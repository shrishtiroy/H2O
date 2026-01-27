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
    Qwen2VL Attention with Heavy-Hitter Oracle (H2O) for KV cache eviction.

    This matches the Qwen2VLAttention interface exactly:
    - Uses position_embeddings (cos, sin) for rotary
    - Uses past_key_values.update() for cache (in-place update)
    - Returns only 2 values: (attn_output, attn_weights)

    Supports Grouped Query Attention (GQA) used in Qwen models.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)

        # Get rope_scaling config for multimodal rotary
        self.rope_scaling = getattr(config, 'rope_scaling', None)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Qwen2VL uses bias=True for QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,  # Note: Qwen2VL uses past_key_values (plural)
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with H2O KV cache eviction.

        Matches Qwen2VLAttention interface - returns only 2 values.
        """
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (bsz, seq_len, num_heads, head_dim) -> (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings (Qwen2VL style)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Try to use Qwen2VL's multimodal rotary
            try:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
                if self.rope_scaling is not None and 'mrope_section' in self.rope_scaling:
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
                    )
                else:
                    # Fallback to standard rotary
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            except Exception:
                # Skip rotary if not compatible
                pass

        # Handle KV cache (Qwen2VL uses Cache object with .update() method)
        if past_key_values is not None:
            # Qwen2VL style: cache is an object with .update() method
            if hasattr(past_key_values, 'update'):
                cache_kwargs = {}
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                # Fallback: tuple-based cache (old style)
                key_states = torch.cat([past_key_values[0], key_states], dim=2)
                value_states = torch.cat([past_key_values[1], value_states], dim=2)

        kv_seq_len = key_states.shape[-2]

        # Repeat KV heads for Grouped Query Attention
        key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights: (bsz, num_heads, q_len, kv_seq_len)
        attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) * self.scaling

        # Apply causal attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        # ===== H2O LOGIC: Apply mask from previous forward =====
        # The mask is computed at the END of each forward and applied at the START of the NEXT forward.
        # On first forward, attention_masks_next is None so no mask is applied.
        if self.attention_masks_next is not None:
            # Ensure mask matches current kv_seq_len
            mask_len = self.attention_masks_next.shape[-1]
            if mask_len == kv_seq_len:
                # Apply mask: keep masked positions, set others to -inf
                attn_weights = attn_weights * self.attention_masks_next + \
                    (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # Softmax (upcast to fp32 for numerical stability)
        attn_weights_softmax = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Safety check: replace any NaN/inf with uniform attention
        if torch.isnan(attn_weights_softmax).any() or torch.isinf(attn_weights_softmax).any():
            attn_weights_softmax = torch.where(
                torch.isnan(attn_weights_softmax) | torch.isinf(attn_weights_softmax),
                torch.ones_like(attn_weights_softmax) / kv_seq_len,
                attn_weights_softmax
            )

        # Apply dropout
        attn_weights_dropped = nn.functional.dropout(
            attn_weights_softmax, p=self.attention_dropout, training=self.training
        )

        # ===== H2O LOGIC: Compute scores and mask for next forward =====
        # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)
        # Sum over batch and query positions: (num_heads, kv_seq_len)
        current_scores_sum = attn_weights_softmax.sum(0).sum(1)

        # Accumulate scores OR initialize budgets
        if self.previous_scores is not None:
            # Subsequent forward: accumulate with previous scores
            if current_scores_sum.shape[-1] > 1:
                current_scores_sum[:, :-1] += self.previous_scores
        else:
            # First forward: compute budgets based on INITIAL sequence length
            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.heavy_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(kv_seq_len)

        dtype_attn = attn_weights_softmax.dtype
        device = attn_weights_softmax.device

        # Store current scores for next forward
        self.previous_scores = current_scores_sum.clone()

        # Create attention mask for NEXT forward
        # Shape: (num_heads, kv_seq_len + 1) - the +1 is for the next token
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1] + 1, dtype=dtype_attn, device=device)

        attn_tokens_all = current_scores_sum.shape[-1]

        if attn_tokens_all > self.cache_budget:
            # More tokens than budget -> need to evict
            if self.recent_budget > 0:
                # Keep recent tokens, zero out the rest
                attn_mask[:, :-self.recent_budget] = 0
                # Select heavy hitters from NON-recent tokens
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                # No recent budget, select from all
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if self.heavy_budget > 0 and selected_set.shape[-1] >= self.heavy_budget:
                # Find top-k heavy hitter tokens
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                # Unmask heavy hitter positions
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        # Store mask for NEXT forward: (1, num_heads, 1, kv_seq_len+1)
        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        # Update previous_scores: zero out evicted tokens
        score_mask = attn_mask[:, :-1]
        if self.recent_budget > 0:
            score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        # ===== END H2O LOGIC =====

        # Compute attention output
        attn_output = torch.matmul(attn_weights_dropped, value_states_repeated)

        # Reshape: (bsz, num_heads, q_len, head_dim) -> (bsz, q_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        # Return only 2 values to match Qwen2VL interface
        if not output_attentions:
            attn_weights_softmax = None

        return attn_output, attn_weights_softmax


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

    This follows the same pattern as convert_kvcache_llama_heavy_recent:
    - Replace attention modules with H2O versions
    - Weights are restored via load_state_dict() called AFTER this function

    Args:
        model: Qwen2-VL or Qwen3-VL model
        config: Model config with heavy_ratio and recent_ratio attributes

    Returns:
        Model with H2O-enabled attention layers (call load_state_dict after!)
    """
    for name, module in reversed(model._modules.items()):
        # Recursively process child modules
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_qwen_heavy_recent(module, config)

        # Replace Qwen3VL vision attention (fused QKV)
        if HAS_QWEN3VL and Qwen3VLAttention is not None and isinstance(module, Qwen3VLAttention):
            model._modules[name] = Qwen3VLVisionAttention_heavy_hitter(config)

        # Replace Qwen2VL attention (separate Q/K/V)
        elif HAS_QWEN2VL and Qwen2VLAttention is not None and isinstance(module, Qwen2VLAttention):
            # Get layer_idx from original module if available
            layer_idx = getattr(module, 'layer_idx', None)
            model._modules[name] = QwenAttention_heavy_hitter(config, layer_idx=layer_idx)

    return model
