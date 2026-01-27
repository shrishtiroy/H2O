"""
H2O (Heavy-Hitter Oracle) implementation for Qwen2-VL and Qwen3-VL models.

This module provides attention classes with KV cache eviction based on the H2O algorithm,
which keeps heavy-hitter tokens (those receiving high cumulative attention) plus recent tokens.
"""

import os
import pdb
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Import Qwen attention classes - handle different transformers versions
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLAttention,
        Qwen2VLRotaryEmbedding,
        apply_rotary_pos_emb_vision,
    )
    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False
    Qwen2VLAttention = None

try:
    from transformers.models.qwen2_vl.modeling_qwen3_vl import Qwen3VLAttention
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False
    Qwen3VLAttention = None


__all__ = ['convert_kvcache_qwen_heavy_recent', 'QwenAttention_heavy_hitter']


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match the number of query heads for Grouped Query Attention.
    From (batch, num_kv_heads, seq_len, head_dim) to (batch, num_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QwenAttention_heavy_hitter(nn.Module):
    """
    Qwen Attention with Heavy-Hitter Oracle (H2O) for KV cache eviction.

    Follows the exact same H2O logic as LlamaAttention_heavy_hitter:
    - On first forward: compute budgets based on initial sequence length
    - On subsequent forwards: accumulate attention scores
    - Evict tokens that are neither heavy-hitters nor recent

    Supports Grouped Query Attention (GQA) used in Qwen models.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 32768)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Qwen uses bias=True for QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings - try to use Qwen's implementation
        try:
            self.rotary_emb = Qwen2VLRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        except:
            # Fallback: will need to handle rotary externally
            self.rotary_emb = None

        # H2O parameters - these persist across forward passes
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
        """Reset H2O state - call between different sequences."""
        self.attention_masks_next = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with H2O KV cache eviction.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Causal attention mask
            position_ids: Position indices for rotary embeddings
            past_key_value: Cached (key, value) tensors
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated KV cache

        Returns:
            attn_output: (batch, seq_len, hidden_size)
            attn_weights: Optional attention weights
            past_key_value: Updated KV cache if use_cache=True
        """
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (bsz, seq_len, num_heads, head_dim) -> (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Determine KV sequence length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary position embeddings if available
        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            # Apply rotary embeddings to Q and K
            # Note: Qwen may use different apply_rotary_pos_emb signature
            try:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            except:
                pass  # Skip rotary if not compatible

        # Concatenate with past KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV heads for Grouped Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights: (bsz, num_heads, q_len, kv_seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply causal attention mask
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # ===== H2O LOGIC: Apply mask from previous forward =====
        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # Softmax (upcast to fp32 for numerical stability)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ===== H2O LOGIC: Compute scores and mask for next forward =====
        # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)
        # Sum over batch and query positions: (num_heads, kv_seq_len)
        current_scores_sum = attn_weights.sum(0).sum(1)

        # Accumulate scores OR initialize budgets
        if not self.previous_scores == None:
            # Subsequent forward: accumulate with previous scores
            current_scores_sum[:, :-1] += self.previous_scores
        else:
            # First forward: compute budgets based on INITIAL sequence length
            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.heavy_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])

        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1, f"H2O requires batch_size=1, got {attn_weights.shape[0]}"

        # Store current scores for next forward
        self.previous_scores = current_scores_sum

        # Create attention mask for NEXT forward
        # Shape: (num_heads, kv_seq_len + 1) - the +1 is for the next token
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1] + 1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]

        if attn_tokens_all > self.cache_budget:
            # More tokens than budget -> need to evict

            if not self.recent_budget == 0:
                # Keep recent tokens, zero out the rest
                attn_mask[:, :-self.recent_budget] = 0
                # Select heavy hitters from NON-recent tokens
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                # No recent budget, select from all
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                # Find top-k heavy hitter tokens
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                # Unmask heavy hitter positions
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        # Store mask for NEXT forward: (1, num_heads, 1, kv_seq_len+1)
        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        # Update previous_scores: zero out evicted tokens
        score_mask = attn_mask[:, :-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        # ===== END H2O LOGIC =====

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape: (bsz, num_heads, q_len, head_dim) -> (bsz, q_len, hidden_size)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3VLVisionAttention_heavy_hitter(nn.Module):
    """
    Qwen3-VL Vision Encoder Attention with H2O.

    This handles the fused QKV format used in Qwen3-VL's vision encoder.
    Vision attention processes image patches with cu_seqlens for variable length.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.num_heads = getattr(config, 'num_attention_heads', getattr(config, 'num_heads', 16))
        self.head_dim = self.dim // self.num_heads

        # Fused QKV projection (Qwen3-VL vision encoder style)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5

        # H2O parameters
        self.heavy_budget_ratio = getattr(config, 'heavy_ratio', 0.1)
        self.recent_budget_ratio = getattr(config, 'recent_ratio', 0.1)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for vision encoder with H2O.

        Args:
            hidden_states: (total_seq_len, hidden_dim)
            cu_seqlens: Cumulative sequence lengths
            position_embeddings: (cos, sin) for rotary
        """
        seq_length = hidden_states.shape[0]

        # Fused QKV projection and reshape
        qkv = self.qkv(hidden_states)
        query_states, key_states, value_states = (
            qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        # Apply rotary embeddings
        if position_embeddings is not None and apply_rotary_pos_emb_vision is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Reshape for batched attention: (1, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Process each image separately based on cu_seqlens
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        q_splits = torch.split(query_states, lengths, dim=2)
        k_splits = torch.split(key_states, lengths, dim=2)
        v_splits = torch.split(value_states, lengths, dim=2)

        attn_outputs = []

        for q, k, v in zip(q_splits, k_splits, v_splits):
            # Reset H2O for each independent image
            self._reset_masks()

            chunk_len = q.shape[2]

            # Compute attention: (1, num_heads, chunk_len, chunk_len)
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

            # Apply H2O mask from previous (if any - usually None for vision)
            if self.attention_masks_next is not None:
                attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

            # Softmax
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

            # H2O score computation (same pattern as LlamaAttention_heavy_hitter)
            current_scores_sum = attn_weights.sum(0).sum(1)  # (num_heads, chunk_len)

            if not self.previous_scores == None:
                current_scores_sum[:, :-1] += self.previous_scores
            else:
                self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
                self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
                self.cache_budget = self.heavy_budget + self.recent_budget
                self.cache_budget_records.append(self.cache_budget)
                self.input_length.append(chunk_len)

            dtype_attn = attn_weights.dtype
            device = attn_weights.device

            self.previous_scores = current_scores_sum
            attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1] + 1).to(dtype_attn).to(device)

            attn_tokens_all = self.previous_scores.shape[-1]

            if attn_tokens_all > self.cache_budget:
                if not self.recent_budget == 0:
                    attn_mask[:, :-self.recent_budget] = 0
                    selected_set = self.previous_scores[:, :-self.recent_budget]
                else:
                    attn_mask[:, :] = 0
                    selected_set = self.previous_scores

                if not self.heavy_budget == 0:
                    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                    attn_mask = attn_mask.scatter(-1, keep_topk, 1)

            self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

            score_mask = attn_mask[:, :-1]
            if self.recent_budget > 0:
                score_mask[:, -self.recent_budget:] = 1
            self.previous_scores = self.previous_scores * score_mask

            # Compute output
            attn_output = torch.matmul(attn_weights, v)
            attn_outputs.append(attn_output)

        # Concatenate all image outputs
        attn_output = torch.cat(attn_outputs, dim=2)

        # Reshape: (1, num_heads, seq_len, head_dim) -> (seq_len, hidden_dim)
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()

        # Output projection
        attn_output = self.proj(attn_output)

        return attn_output


def convert_kvcache_qwen_heavy_recent(model, config):
    """
    Convert Qwen model to use H2O attention layers.

    Args:
        model: Qwen2-VL or Qwen3-VL model
        config: Model config with heavy_ratio and recent_ratio attributes

    Returns:
        Model with H2O-enabled attention layers
    """
    for name, module in reversed(model._modules.items()):
        # Recursively process child modules
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_qwen_heavy_recent(module, config)

        # Replace Qwen3VL vision attention (fused QKV)
        if HAS_QWEN3VL and Qwen3VLAttention is not None and isinstance(module, Qwen3VLAttention):
            new_module = Qwen3VLVisionAttention_heavy_hitter(config)
            # Copy weights
            if hasattr(module, 'qkv'):
                new_module.qkv.weight.data.copy_(module.qkv.weight.data)
                if module.qkv.bias is not None:
                    new_module.qkv.bias.data.copy_(module.qkv.bias.data)
            if hasattr(module, 'proj'):
                new_module.proj.weight.data.copy_(module.proj.weight.data)
                if module.proj.bias is not None:
                    new_module.proj.bias.data.copy_(module.proj.bias.data)
            model._modules[name] = new_module

        # Replace Qwen2VL attention (separate Q/K/V)
        elif HAS_QWEN2VL and Qwen2VLAttention is not None and isinstance(module, Qwen2VLAttention):
            new_module = QwenAttention_heavy_hitter(config)
            # Copy weights from original module
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(module, proj_name) and hasattr(new_module, proj_name):
                    src = getattr(module, proj_name)
                    dst = getattr(new_module, proj_name)
                    dst.weight.data.copy_(src.weight.data)
                    if src.bias is not None and dst.bias is not None:
                        dst.bias.data.copy_(src.bias.data)
            # Copy rotary embedding if present
            if hasattr(module, 'rotary_emb') and hasattr(new_module, 'rotary_emb'):
                if new_module.rotary_emb is not None and module.rotary_emb is not None:
                    new_module.rotary_emb = module.rotary_emb
            model._modules[name] = new_module

    return model
