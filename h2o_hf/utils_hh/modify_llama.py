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
    """Wrapper around LlamaAttention that adds H2O KV cache eviction.

    Prefill (q_len > 1): delegates entirely to the original LlamaAttention,
    ensuring bit-identical results to baseline. Only records the KV cache
    length for budget computation.

    Decode (q_len == 1): uses manual attention with H2O mask and score
    accumulation for KV cache eviction.
    """

    def __init__(self, original_attn: LlamaAttention, config):
        super().__init__()
        # Keep the original attention module for prefill
        self.original_attn = original_attn
        self.config = config
        self.layer_idx = original_attn.layer_idx

        # Copy attributes needed for manual decode attention
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

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

    @property
    def q_proj(self):
        return self.original_attn.q_proj

    @property
    def k_proj(self):
        return self.original_attn.k_proj

    @property
    def v_proj(self):
        return self.original_attn.v_proj

    @property
    def o_proj(self):
        return self.original_attn.o_proj

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(self, hidden_states, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        if q_len > 1:
            # === PREFILL: Delegate to original attention (bit-identical to baseline) ===
            result = self.original_attn(hidden_states, **kwargs)

            # Set H2O budgets based on the KV cache length after prefill
            cache_obj = kwargs.get('past_key_values') or kwargs.get('past_key_value')
            if cache_obj is not None and hasattr(cache_obj, 'get_seq_length'):
                kv_len = cache_obj.get_seq_length(self.layer_idx)
            else:
                kv_len = q_len

            if self.heavy_budget is None:
                self.heavy_budget = max(1, int(self.heavy_budget_ratio * kv_len))
                self.recent_budget = max(1, int(self.recent_budget_ratio * kv_len))
                self.cache_budget = min(self.heavy_budget + self.recent_budget, kv_len)
                self.cache_budget_records.append(self.cache_budget)
                self.input_length.append(kv_len)

            return result

        # === DECODE (q_len == 1): Manual attention with H2O ===

        # Get position embeddings and cache from kwargs
        position_embeddings = kwargs.get('position_embeddings')
        attention_mask = kwargs.get('attention_mask')
        cache_position = kwargs.get('cache_position')
        past_key_value = kwargs.get('past_key_values') or kwargs.get('past_key_value')

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache
        if past_key_value is not None and hasattr(past_key_value, 'update'):
            cache_kwargs = {}
            if position_embeddings is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat KV for GQA
        key_states_rep = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = self.repeat_kv(value_states, self.num_key_value_groups)

        kv_seq_len = key_states_rep.shape[-2]

        # Manual attention
        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.dim() == 4:
                causal_mask = causal_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        # Apply H2O mask if available and sizes match
        if self.attention_masks_next is not None:
            mask_seq_len = self.attention_masks_next.shape[-1]
            if mask_seq_len == kv_seq_len:
                attn_weights = attn_weights * self.attention_masks_next + \
                    (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
            else:
                self._reset_masks()

        # Softmax in float32 for stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # H2O: Accumulate attention scores
        current_scores_sum = attn_weights.sum(0).sum(1)  # (heads, kv_tokens)

        if self.previous_scores is not None:
            # Pad previous scores if KV cache grew
            if current_scores_sum.shape[-1] > self.previous_scores.shape[-1]:
                pad_len = current_scores_sum.shape[-1] - self.previous_scores.shape[-1]
                self.previous_scores = F.pad(self.previous_scores, (0, pad_len), value=0.0)
            current_scores_sum[:, :self.previous_scores.shape[-1]] += self.previous_scores
        else:
            # First decode step — set budgets if not already set during prefill
            if self.heavy_budget is None:
                input_len = current_scores_sum.shape[-1]
                self.heavy_budget = max(1, int(self.heavy_budget_ratio * input_len))
                self.recent_budget = max(1, int(self.recent_budget_ratio * input_len))
                self.cache_budget = min(self.heavy_budget + self.recent_budget, input_len)
                self.cache_budget_records.append(self.cache_budget)
                self.input_length.append(kv_seq_len)

        dtype_attn = attn_weights.dtype
        device_attn = attn_weights.device

        self.previous_scores = current_scores_sum
        attn_mask = torch.ones(
            current_scores_sum.shape[0], current_scores_sum.shape[1] + 1,
            dtype=dtype_attn, device=device_attn)

        attn_tokens_all = self.previous_scores.shape[-1]

        if attn_tokens_all > self.cache_budget:
            if self.recent_budget > 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if self.heavy_budget > 0:
                _, keep_topk = selected_set.topk(
                    k=min(self.heavy_budget, selected_set.shape[-1]),
                    dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        if self.recent_budget > 0:
            score_mask = attn_mask[:, :-1]
            score_mask[:, -self.recent_budget:] = 1
            self.previous_scores = self.previous_scores * score_mask

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states_rep)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, 1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Return format depends on API version
        if NEW_API:
            return attn_output, None
        else:
            past_kv = (key_states, value_states)
            return attn_output, None, past_kv


def convert_kvcache_llama_heavy_recent(model, config):
    """Convert LlamaAttention modules to H2O-enabled attention wrappers."""

    def _convert_recursive(module, config, parent_name=""):
        for name, child in module._modules.items():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if len(list(child.children())) > 0:
                _convert_recursive(child, config, full_name)

            if isinstance(child, LlamaAttention):
                # Wrap with H2O (keeps original as sub-module)
                h2o_attn = LlamaAttention_heavy_hitter(child, config)
                module._modules[name] = h2o_attn

    _convert_recursive(model, config)
    return model
