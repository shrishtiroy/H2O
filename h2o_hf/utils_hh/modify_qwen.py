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

# Try to import ALL_ATTENTION_FUNCTIONS for non-eager attention
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = {}


__all__ = ['convert_kvcache_qwen_heavy_recent', 'QwenAttention_heavy_hitter']


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

        # Repeat KV for GQA (like original)
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        kv_seq_len = key_states_expanded.shape[-2]

        # H2O only activates during autoregressive generation (q_len==1) to avoid
        # numerical divergence from SDPA during prefill
        h2o_active = (q_len == 1 and self.attention_masks_next is not None and
                      self.attention_masks_next.shape[-1] == kv_seq_len)
        h2o_should_track = (q_len == 1 and kv_seq_len >= self.min_seq_for_eviction)

        if not h2o_active and not h2o_should_track:
            # Use SDPA for numerical stability (matches original exactly)
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
            # Manual attention for H2O score tracking and masking
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


def convert_kvcache_qwen_heavy_recent(model, config):
    """Convert Qwen2-VL to use H2O attention."""
    print(f"\n=== H2O Conversion ===")
    
    replaced_count = 0
    
    def should_replace(name, module):
        if 'visual' in name.lower():
            return False
        for cls in QWEN2VL_ATTENTION_CLASSES:
            if cls is not None and isinstance(module, cls):
                return True
        return False
    
    def convert_recursive(parent, parent_name=""):
        nonlocal replaced_count
        
        for name, module in list(parent._modules.items()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if len(list(module.children())) > 0:
                convert_recursive(module, full_name)
            
            if should_replace(full_name, module):
                layer_idx = getattr(module, 'layer_idx', None)
                
                # Create new attention
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
                
                # Match device/dtype
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                new_attn = new_attn.to(device=device, dtype=dtype)
                
                parent._modules[name] = new_attn
                replaced_count += 1
                print(f"  Replaced: {full_name} (layer_idx={layer_idx})")
    
    convert_recursive(model)
    print(f"\nReplaced {replaced_count} attention layers")
    print("=== H2O Conversion Complete ===\n")
    
    return model