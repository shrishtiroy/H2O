"""
StatefulH2OInference — keeps a KV cache alive across browser agent turns.

Each time the browser agent sends a new request it includes the full conversation
history.  This class detects which messages are NEW since the last call, prefills
only those tokens on top of the existing KV cache, generates a response with H2O
eviction active, then physically prunes the KV cache ("hard eviction") so memory
stays bounded regardless of session length.

Usage:
    inf = StatefulH2OInference(model_name="Qwen/Qwen3-VL-2B-Instruct",
                               heavy_ratio=0.1, recent_ratio=0.1)
    response = inf.chat(messages)   # called once per browser-agent turn
"""

import base64
import sys
import os
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.cache_utils import DynamicCache

# Add parent directory so we can import utils_hh
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_hh.modify_qwen import (
    convert_kvcache_qwen_heavy_recent,
    Qwen3VLAttention_heavy_hitter,
    QwenAttention_heavy_hitter,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install qwen-vl-utils: pip install qwen-vl-utils")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    kv_cache: Optional[DynamicCache] = None
    virtual_position: int = 0       # Total tokens ever processed (for RoPE continuity)
    processed_turns: int = 0        # Number of messages already in KV cache
    heavy_budget: Optional[int] = None
    recent_budget: Optional[int] = None


# ---------------------------------------------------------------------------
# Image decoding helpers
# ---------------------------------------------------------------------------

def _decode_image(url: str) -> Image.Image:
    """Decode an image from a data: URI or plain base64 string."""
    if url.startswith("data:"):
        # data:image/png;base64,<payload>
        header, b64 = url.split(",", 1)
    else:
        b64 = url
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGB")


def _extract_images_and_text(content) -> tuple[list[Image.Image], str]:
    """
    Parse an OpenAI-format message content (string or list of parts) into
    (images, combined_text).  Handles the multimodal format that browser-use sends.
    """
    if isinstance(content, str):
        return [], content

    images = []
    text_parts = []
    for part in content:
        ptype = part.get("type", "")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url:
                try:
                    images.append(_decode_image(url))
                except Exception as e:
                    print(f"  [H2O] Warning: failed to decode image: {e}", flush=True)
    return images, "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model_class(model_name: str):
    name = model_name.lower()
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif "qwen2.5" in name or "qwen2_5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StatefulH2OInference:
    """
    Wraps a Qwen3-VL (or 2-VL/2.5-VL) model with H2O attention and maintains
    a stateful KV cache across browser agent turns.

    Thread safety: NOT thread-safe. Use an asyncio lock in the server layer.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        heavy_ratio: float = 0.1,
        recent_ratio: float = 0.1,
        min_seq_for_eviction: int = 500,
        max_pixels: int = 1280 * 1280,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.min_seq_for_eviction = min_seq_for_eviction
        self.max_pixels = max_pixels

        print(f"[H2O Server] Loading {model_name}...", flush=True)
        config = AutoConfig.from_pretrained(model_name)
        config.heavy_ratio = heavy_ratio
        config.recent_ratio = recent_ratio
        config.min_seq_for_eviction = min_seq_for_eviction

        ModelClass = _get_model_class(model_name)
        self.model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.model = convert_kvcache_qwen_heavy_recent(self.model, config)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.session = SessionState()
        print(f"[H2O Server] Ready. heavy={heavy_ratio} recent={recent_ratio} "
              f"min_seq={min_seq_for_eviction}", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """
        Process a browser-agent turn.  `messages` is the FULL conversation history
        (OpenAI format with optional multimodal content).

        Returns the assistant response as a plain string.
        """
        n = len(messages)

        # Detect new session: message count went backwards, or first call
        if n < self.session.processed_turns:
            print(f"[H2O] New session detected (msg count {n} < {self.session.processed_turns}). Resetting.", flush=True)
            self.reset()

        new_messages = messages[self.session.processed_turns:]
        if not new_messages:
            return ""

        print(f"[H2O] Turn {self.session.processed_turns // 2 + 1}: "
              f"processing {len(new_messages)} new messages "
              f"(cache={self._physical_cache_size()} tokens, "
              f"vpos={self.session.virtual_position})", flush=True)

        # --- Build multimodal inputs for new messages only ---
        vision_messages = self._build_vision_messages(new_messages)
        text = self.processor.apply_chat_template(
            vision_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(vision_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        new_token_count = inputs["input_ids"].shape[1]

        # --- Prefill new tokens on top of existing KV cache ---
        if self.session.kv_cache is None:
            self.session.kv_cache = DynamicCache()

        cache_position = torch.arange(
            self.session.virtual_position,
            self.session.virtual_position + new_token_count,
            device=inputs["input_ids"].device,
        )

        with torch.no_grad():
            prefill_out = self.model(
                **inputs,
                past_key_values=self.session.kv_cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )

        self.session.virtual_position += new_token_count
        self.session.processed_turns = n

        # Logits for the last position → first generated token
        next_logits = prefill_out.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # greedy

        # --- Generate response token-by-token ---
        generated_ids = []
        eos_token_id = self.processor.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if tok_id == eos_token_id:
                break

            cache_pos = torch.tensor(
                [self.session.virtual_position],
                device=next_token.device,
            )

            with torch.no_grad():
                step_out = self.model(
                    input_ids=next_token,
                    past_key_values=self.session.kv_cache,
                    cache_position=cache_pos,
                    use_cache=True,
                    return_dict=True,
                )

            self.session.virtual_position += 1
            next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # --- Hard evict: physically prune the KV cache ---
        self._hard_evict()

        response = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True)
        print(f"[H2O] Generated {len(generated_ids)} tokens. "
              f"Cache after eviction: {self._physical_cache_size()} tokens.", flush=True)
        return response

    def reset(self):
        """Clear session state for a new browser task."""
        self.session = SessionState()
        self._reset_h2o_module_state()
        torch.cuda.empty_cache()
        print("[H2O] Session reset.", flush=True)

    def session_status(self) -> dict:
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        return {
            "processed_turns": self.session.processed_turns,
            "virtual_position": self.session.virtual_position,
            "physical_kv_size": self._physical_cache_size(),
            "heavy_budget": self.session.heavy_budget,
            "recent_budget": self.session.recent_budget,
            "gpu_memory_allocated_mb": round(mem_mb, 1),
        }

    # ------------------------------------------------------------------
    # Hard eviction
    # ------------------------------------------------------------------

    def _hard_evict(self):
        """
        After generation, read the H2O survival masks from each attention layer
        and physically remove evicted positions from the KV cache tensors.

        This keeps memory bounded across turns rather than growing indefinitely.
        """
        if self.session.kv_cache is None:
            return

        # Collect per-layer keep masks (union across heads)
        layer_masks = {}
        for module in self.model.modules():
            if isinstance(module, (Qwen3VLAttention_heavy_hitter, QwenAttention_heavy_hitter)):
                if module.attention_masks_next is None:
                    continue
                layer_idx = module.layer_idx
                if layer_idx is None:
                    continue
                # mask shape: (1, num_heads, 1, seq_len+1) — drop the +1 "next" slot
                mask = module.attention_masks_next[0, :, 0, :-1]  # (heads, seq_len)
                keep = mask.any(dim=0)  # (seq_len,) — keep if any head keeps it
                layer_masks[layer_idx] = keep

        if not layer_masks:
            return

        # Global keep = union across all layers
        masks_list = list(layer_masks.values())
        seq_len = masks_list[0].shape[0]
        global_keep = torch.zeros(seq_len, dtype=torch.bool,
                                  device=masks_list[0].device)
        for m in masks_list:
            if m.shape[0] == seq_len:
                global_keep = global_keep | m

        kept = global_keep.sum().item()
        if kept == seq_len:
            return  # Nothing to evict yet

        keep_indices = global_keep.nonzero(as_tuple=True)[0]

        # Physically prune all KV cache layers
        cache = self.session.kv_cache
        num_layers = len(cache.key_cache) if hasattr(cache, 'key_cache') else 0

        if num_layers > 0:
            # Transformers 5.x DynamicCache with key_cache / value_cache lists
            for i in range(num_layers):
                if cache.key_cache[i] is not None and cache.key_cache[i].numel() > 0:
                    k = cache.key_cache[i]   # (batch, heads, seq, head_dim)
                    v = cache.value_cache[i]
                    if k.shape[-2] == seq_len:
                        cache.key_cache[i]   = k[..., keep_indices, :]
                        cache.value_cache[i] = v[..., keep_indices, :]
        else:
            # Fallback: DynamicCache with .layers attribute
            for layer_obj in getattr(cache, 'layers', []):
                if hasattr(layer_obj, 'keys') and layer_obj.keys is not None:
                    k = layer_obj.keys
                    v = layer_obj.values
                    if k.shape[-2] == seq_len:
                        layer_obj.keys   = k[..., keep_indices, :]
                        layer_obj.values = v[..., keep_indices, :]

        # Prune H2O scores to match the new physical positions
        for module in self.model.modules():
            if isinstance(module, (Qwen3VLAttention_heavy_hitter, QwenAttention_heavy_hitter)):
                if module.h2o_scores is not None:
                    s = module.h2o_scores  # (heads, seq_len)
                    if s.shape[-1] == seq_len:
                        module.h2o_scores = s[:, keep_indices]
                # Clear the mask — it will be rebuilt on next decode step
                module.attention_masks_next = None

        # Update budget based on first eviction
        if self.session.heavy_budget is None:
            self.session.heavy_budget = max(1, int(self.heavy_ratio * kept))
            self.session.recent_budget = max(1, int(self.recent_ratio * kept))

        print(f"[H2O] Hard eviction: {seq_len} → {kept} tokens "
              f"({seq_len - kept} evicted)", flush=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_h2o_module_state(self):
        for module in self.model.modules():
            if hasattr(module, '_reset_masks'):
                module._reset_masks()

    def _physical_cache_size(self) -> int:
        if self.session.kv_cache is None:
            return 0
        cache = self.session.kv_cache
        # Try key_cache list (transformers 5.x)
        if hasattr(cache, 'key_cache'):
            for k in cache.key_cache:
                if k is not None and k.numel() > 0:
                    return k.shape[-2]
        # Fallback: layers attr
        for layer_obj in getattr(cache, 'layers', []):
            if hasattr(layer_obj, 'keys') and layer_obj.keys is not None:
                return layer_obj.keys.shape[-2]
        return 0

    def _build_vision_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert OpenAI-format messages (with image_url content parts) into
        the qwen_vl_utils format expected by process_vision_info.
        """
        out = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            # Multimodal content list
            new_content = []
            for part in content:
                ptype = part.get("type", "")
                if ptype == "text":
                    new_content.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url:
                        try:
                            img = _decode_image(url)
                            new_content.append({
                                "type": "image",
                                "image": img,
                                "max_pixels": self.max_pixels,
                            })
                        except Exception as e:
                            print(f"  [H2O] Image decode error: {e}", flush=True)
            out.append({"role": role, "content": new_content})
        return out
