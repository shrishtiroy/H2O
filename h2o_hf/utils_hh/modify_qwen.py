"""
H2O for Qwen2-VL - Version 6

This version exactly matches the Qwen2VLAttention interface and implementation,
only adding H2O eviction logic.
"""

import math
from typing import Optional, Tuple, Callable

import torch
from torch import nn
import torch.nn.functional as F

# Import everything we need from the original
HAS_QWEN2VL = False
QWEN2VL_ATTENTION_CLASSES = []

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLAttention,
        apply_multimodal_rotary_pos_emb,
        eager_attention_forward,
    )
    HAS_QWEN2VL = True
    QWEN2VL_ATTENTION_CLASSES.append(Qwen2VLAttention)
except ImportError:
    Qwen2VLAttention = None
    apply_multimodal_rotary_pos_emb = None
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,  # Match the parameter name exactly
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
        
        # Reset H2O scores at the start of a new sequence
        if cache_position is not None and cache_position[0].item() == 0:
            self.h2o_scores = None

        # QKV projection - exactly as original
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape - use -1 for automatic head count inference like original
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings - exactly as original
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # Update cache - exactly as original
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Use the attention interface like original OR fall back to manual computation
        # For consistency and H2O support, we use manual computation
        
        # Get sequence length
        kv_seq_len = key_states.shape[-2]
        
        # Check if we should use H2O eviction
        use_h2o = kv_seq_len >= self.min_seq_for_eviction
        
        # Manual attention computation (with optional H2O)
        # Repeat KV for GQA
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) * self.scaling

        # Apply attention mask (before softmax)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        # Softmax - use float32 for numerical stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Ensure no NaN from softmax
        if torch.isnan(attn_weights).any():
            print(f"Warning: NaN detected in attention weights after softmax")
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # H2O eviction
        if use_h2o:
            h2o_mask = self._compute_h2o_mask(attn_weights, kv_seq_len)
            if h2o_mask is not None:
                # Apply mask - multiply by 0 for evicted tokens
                attn_weights = attn_weights * h2o_mask.float()
                
                # Renormalize carefully to avoid NaN
                attn_sum = attn_weights.sum(dim=-1, keepdim=True)
                # Ensure we don't divide by zero
                attn_sum = torch.clamp(attn_sum, min=1e-9)
                attn_weights = attn_weights / attn_sum
                
                # Convert back to original dtype after renormalization
                attn_weights = attn_weights.to(query_states.dtype)

        # Dropout
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states_expanded)
        
        # Reshape: (bsz, num_heads, q_len, head_dim) -> (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Output reshape and projection - exactly as original
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _compute_h2o_mask(self, attn_weights, kv_seq_len):
        """Compute H2O eviction mask."""
        dtype = attn_weights.dtype
        device = attn_weights.device
        
        # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)
        # Verify kv_seq_len matches the actual attention weights
        actual_kv_len = attn_weights.shape[-1]
        if actual_kv_len != kv_seq_len:
            print(f"Warning: kv_seq_len mismatch: {kv_seq_len} vs {actual_kv_len}, using actual", file=__import__('sys').stderr)
            kv_seq_len = actual_kv_len
        
        # Aggregate scores across batch and query positions
        current_scores = attn_weights.sum(dim=(0, 2))  # (num_heads, kv_seq_len)
        
        # Debug logging for H2O activation
        heavy_budget = max(1, int(self.heavy_budget_ratio * kv_seq_len))
        recent_budget = max(1, int(self.recent_budget_ratio * kv_seq_len))
        sink_budget = self.sink_token_count
        total_keep = sink_budget + recent_budget + heavy_budget
        
        if kv_seq_len >= self.min_seq_for_eviction and kv_seq_len > total_keep:
            import sys
            print(f"\n[H2O] Layer {self.layer_idx}: seq_len={kv_seq_len}, keep={total_keep} "
                  f"(sink={sink_budget}, recent={recent_budget}, heavy={heavy_budget})", file=sys.stderr)
        
        # Initialize h2o_scores if needed or if sequence length changed
        if self.h2o_scores is None:
            self.h2o_scores = current_scores.detach().clone()
        elif self.h2o_scores.shape[-1] != kv_seq_len:
            # Sequence length changed - reset scores
            self.h2o_scores = current_scores.detach().clone()
        else:
            # Same length - accumulate
            current_scores = current_scores + self.h2o_scores
        
        heavy_budget = max(1, int(self.heavy_budget_ratio * kv_seq_len))
        recent_budget = max(1, int(self.recent_budget_ratio * kv_seq_len))
        sink_budget = self.sink_token_count
        
        # Total tokens to keep
        total_keep = sink_budget + recent_budget + heavy_budget
        
        if kv_seq_len <= total_keep:
            self.h2o_scores = current_scores.detach().clone()
            return None
        
        # Create keep mask for exact kv_seq_len
        keep_mask = torch.zeros(self.num_heads, kv_seq_len, dtype=torch.bool, device=device)
        
        # CRITICAL FIX FOR VLMs: Protect image tokens (usually at the start)
        # For Qwen2-VL, image patches are embedded at the beginning (~400-600 tokens)
        # but we should be smarter about this - only protect tokens that matter
        # Use a smaller protected zone to avoid defeating H2O's purpose
        image_protect_size = max(sink_budget * 2, int(0.2 * kv_seq_len))  # Only protect first 20%
        if kv_seq_len > 3 * image_protect_size:  # Only if there's room
            keep_mask[:, :image_protect_size] = True
            effective_middle_start = image_protect_size
        else:
            # Fallback: just use regular sink budget
            keep_mask[:, :sink_budget] = True
            effective_middle_start = sink_budget
        
        # Keep recent tokens (end)
        if recent_budget > 0:
            keep_mask[:, -recent_budget:] = True
        
        # Keep heavy hitter tokens from middle section
        middle_start = effective_middle_start
        middle_end = kv_seq_len - recent_budget if recent_budget > 0 else kv_seq_len
        
        if middle_end > middle_start and heavy_budget > 0:
            middle_len = middle_end - middle_start
            middle_scores = current_scores[:, middle_start:middle_end]
            k = min(heavy_budget, middle_len)
            if k > 0 and middle_len > 0:
                _, topk_idx = middle_scores.topk(k=k, dim=-1, largest=True)
                keep_mask.scatter_(1, topk_idx + middle_start, True)
        
        # Update scores - only keep scores for tokens we're keeping
        self.h2o_scores = (current_scores * keep_mask.to(current_scores.dtype)).detach().clone()
        
        # Return mask in shape (1, num_heads, 1, kv_seq_len) for broadcasting
        # Ensure shape exactly matches for broadcasting with attention weights
        return keep_mask.unsqueeze(0).unsqueeze(2).to(dtype)


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