"""
Custom SGLang attention backend: FlashInfer for prefill, manual attention
with H2O score tracking for decode.

Registration note:
  SGLang spawns scheduler subprocesses, so this backend must be imported
  inside those subprocesses.  The recommended way is to add the dynamic-
  import hook from ``_patch_attention_registry()`` to SGLang's
  ``attention_registry.py``.  See inference_sglang.py for automatic setup.
"""

import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    register_attention_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class H2OFlashInferAttnBackend(FlashInferAttnBackend):
    """
    FlashInfer backend with H2O attention-score tracking during decode.

    * ``forward_extend`` (prefill) — delegates to FlashInfer unchanged.
    * ``forward_decode`` — when the sequence length exceeds the eviction
      threshold *and* the batch contains a single request, falls back to
      manual QK^T attention so that per-key weights are available for H2O
      score accumulation.  For batch_size > 1 or short sequences, FlashInfer
      is used as normal.
    """

    def __init__(self, model_runner: "ModelRunner", **kwargs):
        super().__init__(model_runner, **kwargs)
        self._model_runner = model_runner

        h2o_cfg = getattr(model_runner, "h2o_config", None) or {}
        self.heavy_ratio: float = float(h2o_cfg.get("heavy_ratio", 0.1))
        self.recent_ratio: float = float(h2o_cfg.get("recent_ratio", 0.1))
        self.min_seq_for_eviction: int = int(
            h2o_cfg.get("min_seq_for_eviction", 500)
        )

        self.h2o_scores: Dict[int, torch.Tensor] = {}
        self.h2o_masks: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    #  Decode – manual attention with H2O scoring                         #
    # ------------------------------------------------------------------ #

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        batch_size = q.shape[0]
        if batch_size != 1:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=False
            )

        seq_len = forward_batch.seq_lens[0].item()
        if seq_len < self.min_seq_for_eviction:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=False
            )

        # --- Manual attention for H2O scoring ---
        num_q_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        gqa_groups = num_q_heads // num_kv_heads

        req_idx = forward_batch.req_pool_indices[0]
        token_indices = forward_batch.req_to_token_pool.req_to_token[
            req_idx, :seq_len
        ]
        k_buf, v_buf = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        K = k_buf[token_indices]  # (seq_len, num_kv_heads, head_dim)
        V = v_buf[token_indices]

        Q = q.view(1, num_q_heads, head_dim).unsqueeze(2)
        # (1, num_q_heads, 1, head_dim)

        K = K.unsqueeze(0).permute(0, 2, 1, 3)  # (1, kv_heads, seq, dim)
        V = V.unsqueeze(0).permute(0, 2, 1, 3)
        if gqa_groups > 1:
            K = K.repeat_interleave(gqa_groups, dim=1)
            V = V.repeat_interleave(gqa_groups, dim=1)

        attn_w = (
            torch.matmul(Q.float(), K.float().transpose(-2, -1))
            * layer.scaling
        )

        layer_id = layer.layer_id
        if layer_id in self.h2o_masks:
            mask = self.h2o_masks[layer_id]
            if mask.shape[-1] == seq_len:
                attn_w = attn_w * mask.float() + (
                    1.0 - mask.float()
                ) * torch.finfo(attn_w.dtype).min

        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32)
        self._update_h2o_state(layer_id, attn_w, seq_len, num_q_heads)

        out = torch.matmul(attn_w.to(V.dtype), V)
        return out.squeeze(2).reshape(1, num_q_heads * head_dim)

    # ------------------------------------------------------------------ #
    #  H2O score tracking                                                 #
    # ------------------------------------------------------------------ #

    def _update_h2o_state(
        self,
        layer_id: int,
        attn_weights: torch.Tensor,
        seq_len: int,
        num_q_heads: int,
    ):
        device = attn_weights.device
        # (1, q_heads, 1, seq_len) → (q_heads, seq_len)
        cur = attn_weights.detach().squeeze(0).squeeze(1)

        if layer_id in self.h2o_scores:
            prev = self.h2o_scores[layer_id]
            if prev.shape[-1] < seq_len:
                pad = torch.zeros(
                    prev.shape[0],
                    seq_len - prev.shape[-1],
                    dtype=prev.dtype,
                    device=device,
                )
                prev = torch.cat([prev, pad], dim=-1)
            if prev.shape[-1] == seq_len:
                cur = cur + prev

        self.h2o_scores[layer_id] = cur.clone()

        heavy_budget = max(1, int(self.heavy_ratio * seq_len))
        recent_budget = max(1, int(self.recent_ratio * seq_len))
        cache_budget = heavy_budget + recent_budget
        next_len = seq_len + 1

        mask = torch.ones(
            num_q_heads, next_len, dtype=torch.float32, device=device
        )
        if seq_len > cache_budget:
            if recent_budget > 0:
                mask[:, :-recent_budget] = 0
                selected = cur[:, :-recent_budget]
            else:
                mask[:, :] = 0
                selected = cur
            if heavy_budget > 0:
                topk = min(heavy_budget, selected.shape[-1])
                _, keep_idx = selected.topk(k=topk, dim=-1, largest=True)
                mask = mask.scatter(-1, keep_idx, 1.0)

        self.h2o_masks[layer_id] = mask.unsqueeze(0).unsqueeze(2)

        if recent_budget > 0:
            score_mask = mask[:, :-1].clone()
            score_mask[:, -recent_budget:] = 1.0
            self.h2o_scores[layer_id] = self.h2o_scores[layer_id] * score_mask

    def reset_h2o_state(self):
        self.h2o_scores.clear()
        self.h2o_masks.clear()


# ====================================================================== #
#  Registration                                                           #
# ====================================================================== #

def _make_h2o_backend(runner: "ModelRunner"):
    """Factory registered under the name ``h2o_flashinfer``."""
    if runner.use_mla_backend:
        raise ValueError("H2O backend does not support MLA models")

    if runner.server_args.speculative_algorithm == "EAGLE":
        if not getattr(runner, "plan_stream_for_flashinfer", None):
            runner.plan_stream_for_flashinfer = torch.cuda.Stream()

    return H2OFlashInferAttnBackend(
        runner, init_new_workspace=runner.init_new_workspace
    )


if "h2o_flashinfer" not in ATTENTION_BACKENDS:
    ATTENTION_BACKENDS["h2o_flashinfer"] = _make_h2o_backend


# ---------------------------------------------------------------------- #
#  Utility: patch SGLang's attention_registry.py so that spawned          #
#  scheduler sub-processes can discover this backend automatically.       #
# ---------------------------------------------------------------------- #

_HOOK_MARKER = "# h2o_dynamic_backend_hook"


def _patch_attention_registry() -> bool:
    """
    Append a small import-hook to the installed ``attention_registry.py``
    so that every newly-spawned SGLang subprocess registers the H2O
    backend.  Safe to call multiple times (idempotent).

    Returns True if the hook was added (or already present), False on error.
    """
    import sglang.srt.layers.attention.attention_registry as reg_mod

    reg_path = reg_mod.__file__
    if reg_path is None:
        return False

    try:
        with open(reg_path, "r") as f:
            source = f.read()
    except OSError:
        return False

    if _HOOK_MARKER in source:
        return True

    hook = f"""
{_HOOK_MARKER}
import os as _os
_h2o_mod = _os.environ.get("SGLANG_H2O_BACKEND_MODULE")
if _h2o_mod:
    import importlib as _il
    try:
        _il.import_module(_h2o_mod)
    except Exception:
        pass
"""

    try:
        with open(reg_path, "a") as f:
            f.write(hook)
        return True
    except OSError:
        return False
