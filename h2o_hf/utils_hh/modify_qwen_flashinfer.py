"""
FlashInfer attention replacement for Qwen2-VL / Qwen2.5-VL / Qwen3-VL.

Replaces the HF attention layers with FlashInfer fused kernels for both prefill
and decode.  No eviction, no scoring — just fast attention with full KV cache.

HF DynamicCache stores K/V as (batch=1, num_kv_heads, seq_len, head_dim) which
maps directly to FlashInfer's HND layout after squeezing the batch dim.

API used:
  prefill: flashinfer.single_prefill_with_kv_cache(q, k, v, kv_layout="HND")
  decode:  flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="HND")

Both handle GQA natively (num_qo_heads != num_kv_heads), so no repeat_kv needed.
"""

import inspect
from typing import Optional, Tuple

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Import Qwen attention classes (same detection as modify_qwen.py)
# ---------------------------------------------------------------------------

HAS_QWEN2VL = False
QWEN2VL_ATTENTION_CLASSES = []

_ATTN_RETURNS_3 = True
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
    _src = inspect.getsource(Qwen2VLDecoderLayer.forward)
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
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLSdpaAttention
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2VLSdpaAttention)
except ImportError:
    pass

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2VLFlashAttention2)
except ImportError:
    pass

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

# ---------------------------------------------------------------------------
# FlashInfer availability check
# ---------------------------------------------------------------------------

HAS_FLASHINFER = False
try:
    import flashinfer
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major > 7 or (major == 7 and minor >= 5):
            HAS_FLASHINFER = True
        else:
            print(f"[FlashInfer] GPU sm{major}{minor} < sm75 — using PyTorch SDPA fallback")
    else:
        HAS_FLASHINFER = True
except ImportError:
    print("[FlashInfer] Not installed — using PyTorch SDPA fallback")


# ---------------------------------------------------------------------------
# FlashInfer attention helpers
# ---------------------------------------------------------------------------

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA when SDPA doesn't support enable_gqa."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _fast_prefill(query_states, key_states, value_states, sm_scale=None):
    """
    Prefill attention — FlashInfer when available, PyTorch SDPA fallback.

    Input:  HF layout (bsz=1, num_qo_heads, q_len, head_dim) for q,
            (bsz=1, num_kv_heads, kv_len, head_dim) for k/v
    Output: HF layout (bsz=1, num_qo_heads, q_len, head_dim)
    """
    if HAS_FLASHINFER:
        q = query_states.squeeze(0).transpose(0, 1).contiguous()
        k = key_states.squeeze(0)
        v = value_states.squeeze(0)
        o = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=True, sm_scale=sm_scale, kv_layout="HND",
        )
        return o.transpose(0, 1).unsqueeze(0)

    num_qo_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    n_rep = num_qo_heads // num_kv_heads
    k_exp = _repeat_kv(key_states, n_rep)
    v_exp = _repeat_kv(value_states, n_rep)
    return torch.nn.functional.scaled_dot_product_attention(
        query_states, k_exp, v_exp, is_causal=True, scale=sm_scale)


def _fast_decode(query_states, key_states, value_states, sm_scale=None):
    """
    Decode attention (single token) — FlashInfer when available, SDPA fallback.

    Input:  HF layout (bsz=1, num_qo_heads, 1, head_dim) for q,
            (bsz=1, num_kv_heads, kv_len, head_dim) for k/v
    Output: HF layout (bsz=1, num_qo_heads, 1, head_dim)
    """
    if HAS_FLASHINFER:
        q = query_states.squeeze(0).squeeze(1)
        k = key_states.squeeze(0)
        v = value_states.squeeze(0)
        o = flashinfer.single_decode_with_kv_cache(
            q, k, v, kv_layout="HND", sm_scale=sm_scale,
        )
        return o.unsqueeze(0).unsqueeze(2)

    num_qo_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    n_rep = num_qo_heads // num_kv_heads
    k_exp = _repeat_kv(key_states, n_rep)
    v_exp = _repeat_kv(value_states, n_rep)
    return torch.nn.functional.scaled_dot_product_attention(
        query_states, k_exp, v_exp, is_causal=False, scale=sm_scale)


# ---------------------------------------------------------------------------
# Qwen3-VL FlashInfer attention wrapper
# ---------------------------------------------------------------------------

class FlashInferQwen3VLAttention(nn.Module):
    """
    Wraps Qwen3VLTextAttention, replacing the attention kernel with FlashInfer
    for both prefill and decode.  Preserves q_norm/k_norm/rotary/cache logic.
    """

    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = original_attn.layer_idx
        self.head_dim = original_attn.head_dim
        self.num_heads = original_attn.q_proj.out_features // self.head_dim
        self.num_key_value_heads = original_attn.k_proj.out_features // self.head_dim
        self.hidden_size = original_attn.o_proj.in_features
        self.scaling = original_attn.scaling

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
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        if q_len > 1:
            attn_output = _fast_prefill(
                query_states, key_states, value_states, sm_scale=self.scaling)
        else:
            attn_output = _fast_decode(
                query_states, key_states, value_states, sm_scale=self.scaling)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.original_attn.o_proj(attn_output)

        return attn_output, None


# ---------------------------------------------------------------------------
# Qwen2-VL / Qwen2.5-VL FlashInfer attention wrapper
# ---------------------------------------------------------------------------

class FlashInferQwen2VLAttention(nn.Module):
    """
    Wraps Qwen2VLAttention (or Qwen2.5-VL variants), replacing the attention
    kernel with FlashInfer for both prefill and decode.
    """

    def __init__(self, original_attn, config):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = getattr(original_attn, 'layer_idx', None)
        self.head_dim = getattr(original_attn, 'head_dim',
                                config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim ** -0.5
        self.rope_scaling = getattr(config, 'rope_scaling', {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.original_attn.q_proj(hidden_states)
        key_states = self.original_attn.k_proj(hidden_states)
        value_states = self.original_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        if apply_multimodal_rotary_pos_emb is not None:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        if q_len > 1:
            attn_output = _fast_prefill(
                query_states, key_states, value_states, sm_scale=self.scaling)
        else:
            attn_output = _fast_decode(
                query_states, key_states, value_states, sm_scale=self.scaling)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.original_attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if _ATTN_RETURNS_3:
            return attn_output, None, past_key_value
        else:
            return attn_output, None


# ---------------------------------------------------------------------------
# Conversion entry point
# ---------------------------------------------------------------------------

def convert_attention_to_flashinfer(model):
    """
    Walk the model and replace all text-attention layers with FlashInfer-backed
    wrappers.  Vision encoder attention is left untouched.
    """
    print("\n=== FlashInfer Attention Conversion ===")
    replaced_count = 0

    def _get_replacement_type(name, module):
        if 'visual' in name.lower():
            return None
        for cls in QWEN3VL_ATTENTION_CLASSES:
            if cls is not None and isinstance(module, cls):
                return 'qwen3vl'
        for cls in QWEN2VL_ATTENTION_CLASSES:
            if cls is not None and isinstance(module, cls):
                return 'qwen2vl'
        return None

    def _convert_recursive(parent, parent_name=""):
        nonlocal replaced_count

        for name, module in list(parent._modules.items()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            if len(list(module.children())) > 0:
                _convert_recursive(module, full_name)

            rtype = _get_replacement_type(full_name, module)
            if rtype is None:
                continue

            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype

            if rtype == 'qwen3vl':
                new_attn = FlashInferQwen3VLAttention(module)
            else:
                config = model.config
                if hasattr(config, 'text_config'):
                    config = config.text_config
                new_attn = FlashInferQwen2VLAttention(module, config)

            new_attn = new_attn.to(device=device, dtype=dtype)
            parent._modules[name] = new_attn
            replaced_count += 1
            layer_idx = getattr(module, 'layer_idx', '?')
            print(f"  FlashInfer: {full_name} (layer_idx={layer_idx})")

    _convert_recursive(model)
    print(f"\nReplaced {replaced_count} attention layers")
    if HAS_FLASHINFER:
        print(f"  Backend: FlashInfer fused kernels (prefill + decode)")
    else:
        print(f"  Backend: PyTorch SDPA fallback (FlashInfer unavailable)")
    print("=== Attention Conversion Complete ===\n")

    return model
