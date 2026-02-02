import os
import pdb
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Callable

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

# Check transformers version for API compatibility
import transformers
TRANSFORMERS_VERSION = tuple(int(x) for x in transformers.__version__.split('.')[:2])
NEW_API = TRANSFORMERS_VERSION >= (4, 45)  # New API in transformers >= 4.45

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']


class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with H2O"""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.scaling = self.head_dim ** -0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=getattr(config, 'attention_bias', False))

        # H2O parameters
        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.attention_masks_next = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    def _reset_masks(self):
        self.attention_masks_next = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Legacy parameters for older transformers
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values (new Cache API)
        if past_key_values is not None:
            # New cache API
            cache_kwargs = {}
            if position_embeddings is not None:
                cache_kwargs = {"sin": position_embeddings[1], "cos": position_embeddings[0], "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        elif past_key_value is not None:
            # Legacy tuple API
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Repeat KV for GQA
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        kv_seq_len = key_states.shape[-2]

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.dim() == 4:
                causal_mask = causal_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        # Apply H2O mask if available and sizes match
        if self.attention_masks_next is not None:
            # Check if mask size matches current kv_seq_len
            mask_seq_len = self.attention_masks_next.shape[-1]
            if mask_seq_len == kv_seq_len:
                attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
            else:
                # Reset H2O state if sequence length changed (new generation)
                self._reset_masks()

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # H2O: Accumulate attention scores
        current_scores_sum = attn_weights.sum(0).sum(1)  # (heads, k-tokens)

        if self.previous_scores is not None:
            if current_scores_sum.shape[-1] > 1:
                current_scores_sum[:, :-1] += self.previous_scores[:, :current_scores_sum.shape[-1]-1]
        else:
            # Ensure minimum budgets of at least 1 token each (or total input if smaller)
            input_len = current_scores_sum.shape[-1]
            self.heavy_budget = max(1, int(self.heavy_budget_ratio * input_len))
            self.recent_budget = max(1, int(self.recent_budget_ratio * input_len))
            # Ensure total budget doesn't exceed input length
            self.cache_budget = min(self.heavy_budget + self.recent_budget, input_len)
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])

        dtype_attn_weights = attn_weights.dtype
        attn_weights_device = attn_weights.device

        self.previous_scores = current_scores_sum
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1] + 1).to(dtype_attn_weights).to(attn_weights_device)

        attn_tokens_all = self.previous_scores.shape[-1]

        if attn_tokens_all > self.cache_budget:
            if self.recent_budget > 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if self.heavy_budget > 0:
                _, keep_topk = selected_set.topk(k=min(self.heavy_budget, selected_set.shape[-1]), dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        if self.recent_budget > 0:
            score_mask = attn_mask[:, :-1]
            score_mask[:, -self.recent_budget:] = 1
            self.previous_scores = self.previous_scores * score_mask

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Return format depends on API version
        if NEW_API:
            return attn_output, attn_weights if output_attentions else None
        else:
            past_kv = (key_states, value_states) if use_cache else None
            return attn_output, attn_weights if output_attentions else None, past_kv



def convert_kvcache_llama_heavy_recent(model, config):
    """Convert LlamaAttention modules to H2O-enabled attention."""

    def _convert_recursive(module, config, parent_name=""):
        for name, child in module._modules.items():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if len(list(child.children())) > 0:
                _convert_recursive(child, config, full_name)

            if isinstance(child, LlamaAttention):
                # Get layer_idx from the original attention module
                layer_idx = getattr(child, 'layer_idx', 0)

                # Create H2O attention
                h2o_attn = LlamaAttention_heavy_hitter(config, layer_idx=layer_idx)

                # Copy weights from original attention
                h2o_attn.q_proj.weight.data = child.q_proj.weight.data.clone()
                h2o_attn.k_proj.weight.data = child.k_proj.weight.data.clone()
                h2o_attn.v_proj.weight.data = child.v_proj.weight.data.clone()
                h2o_attn.o_proj.weight.data = child.o_proj.weight.data.clone()

                # Copy biases if they exist
                if hasattr(child.q_proj, 'bias') and child.q_proj.bias is not None:
                    h2o_attn.q_proj.bias.data = child.q_proj.bias.data.clone()
                if hasattr(child.k_proj, 'bias') and child.k_proj.bias is not None:
                    h2o_attn.k_proj.bias.data = child.k_proj.bias.data.clone()
                if hasattr(child.v_proj, 'bias') and child.v_proj.bias is not None:
                    h2o_attn.v_proj.bias.data = child.v_proj.bias.data.clone()
                if hasattr(child.o_proj, 'bias') and child.o_proj.bias is not None:
                    h2o_attn.o_proj.bias.data = child.o_proj.bias.data.clone()

                # Move to same device and dtype
                h2o_attn = h2o_attn.to(child.q_proj.weight.device).to(child.q_proj.weight.dtype)

                module._modules[name] = h2o_attn

    _convert_recursive(model, config)
    return model

