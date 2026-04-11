"""
H2O for Qwen2-VL - Version 6

This version exactly matches the Qwen2VLAttention interface and implementation,
only adding H2O eviction logic.
"""

import math
import inspect
from typing import Optional, Tuple, Callable

import torch
from torch import nn
import torch.nn.functional as F

# Import everything we need from the original
HAS_QWEN2VL = False
QWEN2VL_ATTENTION_CLASSES = []

# Auto-detect how many return values the decoder layer expects from self_attn
_ATTN_RETURNS_3 = True  # default: 3 values (attn_output, attn_weights, past_key_value)
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
    _src = inspect.getsource(Qwen2VLDecoderLayer.forward)
    # If the decoder unpacks only 2 values, we should return 2
    if 'self_attn_weights, present_key_value = self.self_attn' not in _src:
        _ATTN_RETURNS_3 = False
except Exception:
    pass

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
except ImportError:
    apply_multimodal_rotary_pos_emb = None

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
    HAS_QWEN2VL = True
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2VLAttention)
except ImportError:
    Qwen2VLAttention = None

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import eager_attention_forward
except ImportError:
    eager_attention_forward = None

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLSdpaAttention
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2VLSdpaAttention)
except ImportError:
    pass

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2VLFlashAttention2)
except ImportError:
    pass

# Qwen2.5-VL support (same interface as Qwen2-VL, different class names)
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2_5_VLAttention)
except ImportError:
    pass

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLSdpaAttention
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2_5_VLSdpaAttention)
except ImportError:
    pass

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2_5_VLFlashAttention2)
except ImportError:
    pass

# Qwen3-VL support (wrapper approach: different interface — q_norm/k_norm, standard rotary, 2 return values)
QWEN3VL_ATTENTION_CLASSES = []
apply_rotary_pos_emb_qwen3vl = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
    QWEN3VL_ATTENTION_CLASSES.append(Qwen3VLTextAttention)
except ImportError:
    Qwen3VLTextAttention = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as _qwen3vl_rope
    apply_rotary_pos_emb_qwen3vl = _qwen3vl_rope
except ImportError:
    pass

# Try to import ALL_ATTENTION_FUNCTIONS for non-eager attention
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = {}


__all__ = ['convert_kvcache_qwen_heavy_recent', 'QwenAttention_heavy_hitter',
           'Qwen3VLAttention_heavy_hitter']

# FlashInfer support — optional fast attention backend (requires sm75+, i.e. Turing or newer)
HAS_FLASHINFER = False
try:
    import flashinfer
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major > 7 or (major == 7 and minor >= 5):
            HAS_FLASHINFER = True
        else:
            print(f"[H2O] FlashInfer skipped: GPU sm{major}{minor} < sm75 (Turing minimum)")
    else:
        HAS_FLASHINFER = True  # defer runtime check
except ImportError:
    pass


def _flashinfer_attention(query_states, key_states, value_states, causal=True, sm_scale=None):
    """
    FlashInfer attention for both prefill and decode. Handles GQA natively
    so callers should pass the *un-expanded* KV tensors (no repeat_kv).

    Input layout  (standard HF):  (bsz=1, num_heads, seq_len, head_dim)
    Output layout (standard HF):  (bsz=1, num_heads, q_len,  head_dim)
    """
    # HF BNHD → FlashInfer NHD  (squeeze batch, swap seq/head axes)
    q = query_states.squeeze(0).transpose(0, 1).contiguous()
    k = key_states.squeeze(0).transpose(0, 1).contiguous()
    v = value_states.squeeze(0).transpose(0, 1).contiguous()

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, sm_scale=sm_scale, kv_layout="NHD",
    )

    return o.transpose(0, 1).unsqueeze(0)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QwenAttention_heavy_hitter(nn.Module):
    """
    Qwen2-VL Attention with H2O - matches original implementation exactly.
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
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        self.is_causal = True
        
        # Match original scaling
        self.scaling = self.head_dim ** -0.5
        
        # Sliding window (if applicable)
        self.sliding_window = getattr(config, 'sliding_window', None)
        
        # Rope scaling - store the full dict, accessed as self.rope_scaling["mrope_section"]
        self.rope_scaling = getattr(config, 'rope_scaling', {})

        # Projections - match original exactly
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # H2O parameters
        self.heavy_budget_ratio = getattr(config, 'heavy_ratio', 0.1)
        self.recent_budget_ratio = getattr(config, 'recent_ratio', 0.1)
        self.sink_token_count = getattr(config, 'sink_token_count', 4)
        self.min_seq_for_eviction = getattr(config, 'min_seq_for_eviction', 1024)
        
        # H2O state tracking - per layer, per sequence
        self.h2o_scores = None
        self.attention_masks_next = None  # Mask to apply on the NEXT forward pass
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,  # Match the parameter name exactly (singular)
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass - matches Qwen2VLAttention.forward exactly.
        Returns: (attn_output, attn_weights)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Reset H2O state at the start of a new sequence
        if cache_position is not None and cache_position[0].item() == 0:
            self.h2o_scores = None
            self.attention_masks_next = None
            self.heavy_budget = None
            self.recent_budget = None
            self.cache_budget = None

        # QKV projection - exactly as original
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # Update cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        kv_seq_len = key_states.shape[-2]

        # H2O only activates during autoregressive generation (q_len==1) to avoid
        # numerical divergence from SDPA during prefill
        h2o_active = (q_len == 1 and self.attention_masks_next is not None and
                      self.attention_masks_next.shape[-1] == kv_seq_len)
        h2o_should_track = (q_len == 1 and kv_seq_len >= self.min_seq_for_eviction)

        if not h2o_active and not h2o_should_track:
            if HAS_FLASHINFER and bsz == 1 and query_states.is_cuda:
                # FlashInfer handles GQA natively — no repeat_kv needed
                attn_output = _flashinfer_attention(
                    query_states, key_states, value_states,
                    causal=True, sm_scale=self.scaling,
                )
            else:
                # Fallback: SDPA with explicit GQA expansion
                key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
                value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

                causal_mask = attention_mask
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, :kv_seq_len]

                if query_states.device.type == "cuda" and causal_mask is not None:
                    query_states = query_states.contiguous()
                    key_states_expanded = key_states_expanded.contiguous()
                    value_states_expanded = value_states_expanded.contiguous()

                is_causal = True if causal_mask is None and q_len > 1 else False

                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded,
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
            attn_weights = None
        else:
            # Manual attention for H2O score tracking — needs GQA-expanded KV
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(
                query_states.float(), key_states_expanded.float().transpose(2, 3)
            ) * self.scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :kv_seq_len]
                attn_weights = attn_weights + causal_mask.float()

            # Apply H2O mask from PREVIOUS step
            if h2o_active:
                attn_weights = attn_weights * self.attention_masks_next.float() + \
                    (1 - self.attention_masks_next.float()) * torch.finfo(attn_weights.dtype).min

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            # Update H2O state for next step
            if h2o_should_track:
                self._update_h2o_state(attn_weights, kv_seq_len)

            attn_weights = attn_weights.to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states_expanded)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if _ATTN_RETURNS_3:
            return attn_output, attn_weights, past_key_value
        else:
            return attn_output, attn_weights

    def _update_h2o_state(self, attn_weights, kv_seq_len):
        """
        Update H2O scores and compute attention mask for NEXT forward pass.
        Follows the LLaMA H2O simulation pattern: mask is computed now but
        applied in the next step.
        """
        device = attn_weights.device

        # Accumulate attention scores: sum over batch and query dims
        # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)
        current_scores_sum = attn_weights.detach().sum(0).sum(1)  # (num_heads, kv_seq_len)

        # Accumulate with previous scores
        if self.h2o_scores is not None:
            if self.h2o_scores.shape[-1] < kv_seq_len:
                # Growing sequence - pad previous scores
                pad_size = kv_seq_len - self.h2o_scores.shape[-1]
                padded = torch.zeros(self.h2o_scores.shape[0], pad_size,
                                     dtype=self.h2o_scores.dtype, device=device)
                self.h2o_scores = torch.cat([self.h2o_scores, padded], dim=-1)
            if self.h2o_scores.shape[-1] == kv_seq_len:
                current_scores_sum[:, :self.h2o_scores.shape[-1]] += self.h2o_scores
        else:
            # First time - initialize budgets
            input_len = kv_seq_len
            self.heavy_budget = max(1, int(self.heavy_budget_ratio * input_len))
            self.recent_budget = max(1, int(self.recent_budget_ratio * input_len))
            self.cache_budget = self.heavy_budget + self.recent_budget

        self.h2o_scores = current_scores_sum.clone()

        # Build mask for NEXT step (size = kv_seq_len + 1 for next token)
        attn_tokens_all = kv_seq_len
        next_len = attn_tokens_all + 1

        attn_mask = torch.ones(current_scores_sum.shape[0], next_len,
                               dtype=attn_weights.dtype, device=device)

        if attn_tokens_all > self.cache_budget:
            # Zero out non-recent, non-heavy-hitter positions
            if self.recent_budget > 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.h2o_scores[:, :-self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = self.h2o_scores

            if self.heavy_budget > 0:
                k = min(self.heavy_budget, selected_set.shape[-1])
                _, keep_topk = selected_set.topk(k=k, dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)  # (1, heads, 1, next_len)

        # Update scores: zero out evicted token scores but keep recent scores
        if self.recent_budget > 0:
            score_mask = attn_mask[:, :-1].clone()
            score_mask[:, -self.recent_budget:] = 1
            self.h2o_scores = self.h2o_scores * score_mask


class Qwen3VLAttention_heavy_hitter(nn.Module):
    """
    Wrapper around Qwen3VLTextAttention that adds H2O KV cache eviction.

    Prefill (q_len > 1): delegates entirely to the original Qwen3VLTextAttention,
    ensuring bit-identical results to baseline.

    Decode (q_len == 1): replicates q_norm/k_norm + rotary + manual attention
    with H2O score accumulation and masking.

    Returns 2 values (attn_output, None) matching Qwen3VLTextDecoderLayer expectation.
    """

    def __init__(self, original_attn, config):
        super().__init__()
        self.original_attn = original_attn
        self.config = config
        self.layer_idx = original_attn.layer_idx

        # Derive dimensions from the projection layers — avoids config nesting issues
        # (Qwen3-VL has a nested text_config; projections are always concrete)
        self.head_dim = original_attn.head_dim
        self.num_heads = original_attn.q_proj.out_features // self.head_dim
        self.num_key_value_heads = original_attn.k_proj.out_features // self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = original_attn.o_proj.in_features
        self.scaling = original_attn.scaling

        # H2O parameters — set on the main config by load_model()
        self.heavy_budget_ratio = getattr(config, 'heavy_ratio', 0.1)
        self.recent_budget_ratio = getattr(config, 'recent_ratio', 0.1)
        self.min_seq_for_eviction = getattr(config, 'min_seq_for_eviction', 1024)

        # H2O state
        self.h2o_scores = None
        self.attention_masks_next = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None

    def _reset_masks(self):
        self.h2o_scores = None
        self.attention_masks_next = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Reset state at the start of a new sequence
        if cache_position is not None and cache_position[0].item() == 0:
            self._reset_masks()

        if q_len > 1:
            if HAS_FLASHINFER and bsz == 1 and hidden_states.is_cuda:
                # FlashInfer prefill — replicate QKV path, swap attention kernel
                cos, sin = position_embeddings
                hidden_shape = (bsz, q_len, -1, self.head_dim)

                query_states = self.original_attn.q_norm(
                    self.original_attn.q_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                key_states = self.original_attn.k_norm(
                    self.original_attn.k_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                value_states = (
                    self.original_attn.v_proj(hidden_states)
                    .view(hidden_shape).transpose(1, 2)
                )

                if apply_rotary_pos_emb_qwen3vl is not None:
                    query_states, key_states = apply_rotary_pos_emb_qwen3vl(
                        query_states, key_states, cos, sin)

                if past_key_values is not None:
                    cache_kwargs = {
                        "sin": sin, "cos": cos, "cache_position": cache_position,
                    }
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs)

                attn_output = _flashinfer_attention(
                    query_states, key_states, value_states,
                    causal=True, sm_scale=self.scaling,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.original_attn.o_proj(attn_output)
                result = (attn_output, None)
            else:
                # Fallback: delegate to original attention
                result = self.original_attn(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            # Set H2O budgets from the KV length after prefill
            if self.heavy_budget is None and past_key_values is not None:
                try:
                    kv_len = past_key_values.get_seq_length(self.layer_idx)
                except Exception:
                    kv_len = q_len
                self.heavy_budget = max(1, int(self.heavy_budget_ratio * kv_len))
                self.recent_budget = max(1, int(self.recent_budget_ratio * kv_len))
                self.cache_budget = self.heavy_budget + self.recent_budget
            return result

        # DECODE (q_len == 1): manual attention with H2O
        cos, sin = position_embeddings

        # Project + apply q_norm and k_norm (Qwen3-VL specific)
        hidden_shape = (bsz, 1, -1, self.head_dim)
        query_states = self.original_attn.q_norm(
            self.original_attn.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)  # (bsz, num_heads, 1, head_dim)
        key_states = self.original_attn.k_norm(
            self.original_attn.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)  # (bsz, num_kv_heads, 1, head_dim)
        value_states = (
            self.original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )  # (bsz, num_kv_heads, 1, head_dim)

        # Apply standard rotary embeddings
        if apply_rotary_pos_emb_qwen3vl is not None:
            query_states, key_states = apply_rotary_pos_emb_qwen3vl(
                query_states, key_states, cos, sin)

        # Update KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        # Expand KV for GQA
        key_states_exp = repeat_kv(key_states, self.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, self.num_key_value_groups)

        kv_seq_len = key_states_exp.shape[-2]

        # Manual attention in float32
        attn_weights = torch.matmul(
            query_states.float(), key_states_exp.float().transpose(2, 3)
        ) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask.float()

        # Apply H2O mask from previous step
        h2o_active = (self.attention_masks_next is not None and
                      self.attention_masks_next.shape[-1] == kv_seq_len)
        if h2o_active:
            attn_weights = (attn_weights * self.attention_masks_next.float() +
                            (1 - self.attention_masks_next.float()) *
                            torch.finfo(attn_weights.dtype).min)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        # H2O: track scores and compute mask for next step
        h2o_should_track = kv_seq_len >= self.min_seq_for_eviction
        if h2o_should_track:
            self._update_h2o_state(attn_weights, kv_seq_len)

        attn_weights_cast = attn_weights.to(query_states.dtype)
        attn_output = torch.matmul(attn_weights_cast, value_states_exp)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, 1, self.hidden_size)
        attn_output = self.original_attn.o_proj(attn_output)

        return attn_output, None

    def _update_h2o_state(self, attn_weights, kv_seq_len):
        """Accumulate attention scores and build eviction mask for next step."""
        device = attn_weights.device
        current_scores = attn_weights.detach().sum(0).sum(1)  # (num_heads, kv_seq_len)

        if self.h2o_scores is not None:
            if self.h2o_scores.shape[-1] < kv_seq_len:
                pad = kv_seq_len - self.h2o_scores.shape[-1]
                self.h2o_scores = torch.cat(
                    [self.h2o_scores,
                     torch.zeros(self.h2o_scores.shape[0], pad,
                                 dtype=self.h2o_scores.dtype, device=device)],
                    dim=-1)
            if self.h2o_scores.shape[-1] == kv_seq_len:
                current_scores = current_scores + self.h2o_scores
        else:
            # First decode step: initialize budgets if not set during prefill
            if self.heavy_budget is None:
                self.heavy_budget = max(1, int(self.heavy_budget_ratio * kv_seq_len))
                self.recent_budget = max(1, int(self.recent_budget_ratio * kv_seq_len))
                self.cache_budget = self.heavy_budget + self.recent_budget

        self.h2o_scores = current_scores.clone()

        # Build mask for next step (size = kv_seq_len + 1 for the next token)
        next_len = kv_seq_len + 1
        attn_mask = torch.ones(current_scores.shape[0], next_len,
                               dtype=attn_weights.dtype, device=device)

        if kv_seq_len > self.cache_budget:
            if self.recent_budget > 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.h2o_scores[:, :-self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = self.h2o_scores

            if self.heavy_budget > 0:
                k = min(self.heavy_budget, selected_set.shape[-1])
                _, keep_topk = selected_set.topk(k=k, dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)

        if self.recent_budget > 0:
            score_mask = attn_mask[:, :-1].clone()
            score_mask[:, -self.recent_budget:] = 1
            self.h2o_scores = self.h2o_scores * score_mask


def convert_kvcache_qwen_heavy_recent(model, config):
    """Convert Qwen2-VL / Qwen2.5-VL / Qwen3-VL to use H2O attention."""
    print(f"\n=== H2O Conversion ===")

    replaced_count = 0

    def get_replacement(name, module):
        """Return (replacement_type, layer_idx) or None if should not replace."""
        if 'visual' in name.lower():
            return None
        # Qwen3-VL: use wrapper approach (preserves q_norm/k_norm/rotary correctly)
        for cls in QWEN3VL_ATTENTION_CLASSES:
            if cls is not None and isinstance(module, cls):
                return ('qwen3vl', getattr(module, 'layer_idx', None))
        # Qwen2-VL / Qwen2.5-VL: use replacement approach
        for cls in QWEN2VL_ATTENTION_CLASSES:
            if cls is not None and isinstance(module, cls):
                return ('qwen2vl', getattr(module, 'layer_idx', None))
        return None

    def convert_recursive(parent, parent_name=""):
        nonlocal replaced_count

        for name, module in list(parent._modules.items()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            if len(list(module.children())) > 0:
                convert_recursive(module, full_name)

            replacement = get_replacement(full_name, module)
            if replacement is None:
                continue

            replacement_type, layer_idx = replacement
            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype

            if replacement_type == 'qwen3vl':
                # Wrapper: keep original as sub-module, no weight copying needed
                new_attn = Qwen3VLAttention_heavy_hitter(module, config)
                new_attn = new_attn.to(device=device, dtype=dtype)
                parent._modules[name] = new_attn
                replaced_count += 1
                print(f"  Wrapped (Qwen3-VL): {full_name} (layer_idx={layer_idx})")

            else:  # qwen2vl
                new_attn = QwenAttention_heavy_hitter(config, layer_idx=layer_idx)

                # Copy weights
                with torch.no_grad():
                    new_attn.q_proj.weight.copy_(module.q_proj.weight)
                    new_attn.q_proj.bias.copy_(module.q_proj.bias)
                    new_attn.k_proj.weight.copy_(module.k_proj.weight)
                    new_attn.k_proj.bias.copy_(module.k_proj.bias)
                    new_attn.v_proj.weight.copy_(module.v_proj.weight)
                    new_attn.v_proj.bias.copy_(module.v_proj.bias)
                    new_attn.o_proj.weight.copy_(module.o_proj.weight)

                new_attn = new_attn.to(device=device, dtype=dtype)
                parent._modules[name] = new_attn
                replaced_count += 1
                print(f"  Replaced (Qwen2-VL): {full_name} (layer_idx={layer_idx})")

    convert_recursive(model)
    print(f"\nReplaced {replaced_count} attention layers")
    if HAS_FLASHINFER:
        print(f"FlashInfer: ENABLED (prefill + non-H2O decode use FlashInfer kernels)")
    else:
        print(f"FlashInfer: not available (using PyTorch SDPA fallback)")
    print("=== H2O Conversion Complete ===\n")

    return model