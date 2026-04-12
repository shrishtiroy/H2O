"""
StatefulFlashInferInference — keeps a KV cache alive across browser agent turns.

Each time the browser agent sends a new request it includes the full conversation
history.  This class detects which messages are NEW since the last call, prefills
only those tokens on top of the existing KV cache, and generates a response using
FlashInfer fused attention kernels (both prefill and decode).

No eviction is performed — the KV cache grows with the conversation.  FlashInfer
handles GQA natively and provides fused CUDA kernels for both prefill and decode.

Usage:
    inf = StatefulFlashInferInference(model_name="Qwen/Qwen3-VL-4B-Instruct")
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_hh.modify_qwen_flashinfer import convert_attention_to_flashinfer

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
    virtual_position: int = 0
    processed_turns: int = 0


# ---------------------------------------------------------------------------
# Image decoding helpers
# ---------------------------------------------------------------------------

def _decode_image(url: str) -> Image.Image:
    """Decode an image from a data: URI or plain base64 string."""
    if url.startswith("data:"):
        header, b64 = url.split(",", 1)
    else:
        b64 = url
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGB")


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

class StatefulFlashInferInference:
    """
    Wraps a Qwen3-VL (or 2-VL/2.5-VL) model with FlashInfer attention and
    maintains a stateful KV cache across browser agent turns.

    Thread safety: NOT thread-safe. Use an asyncio lock in the server layer.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_pixels: int = 1280 * 1280,
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.max_pixels = max_pixels

        print(f"[FlashInfer Server] Loading {model_name}"
              f"{' (4-bit quantized)' if load_in_4bit else ''}...", flush=True)

        ModelClass = _get_model_class(model_name)

        load_kwargs = dict(device_map=device_map)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["dtype"] = torch.float16

        self.model = ModelClass.from_pretrained(model_name, **load_kwargs)
        self.model = convert_attention_to_flashinfer(self.model)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.session = SessionState()
        print(f"[FlashInfer Server] Ready.", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], max_new_tokens: int = 4096,
             temperature: float = 0.0) -> str:
        """
        Process a browser-agent turn.  `messages` is the FULL conversation history
        (OpenAI format with optional multimodal content).

        Returns the assistant response as a plain string.
        """
        n = len(messages)

        if n < self.session.processed_turns:
            print(f"[FI] New session detected (msg count {n} < "
                  f"{self.session.processed_turns}). Resetting.", flush=True)
            self.reset()

        new_messages = messages[self.session.processed_turns:]
        if not new_messages:
            if n > 0 and messages[-1].get("role") == "user":
                print(f"[FI] Retry detected — full reset and re-process", flush=True)
                self.reset()
                new_messages = messages
            if not new_messages:
                return ""

        print(f"[FI] Turn {self.session.processed_turns // 2 + 1}: "
              f"processing {len(new_messages)} new messages "
              f"(cache={self._physical_cache_size()} tokens, "
              f"vpos={self.session.virtual_position}, "
              f"temp={temperature})", flush=True)

        # --- Build multimodal inputs for new messages only ---
        vision_messages = self._build_vision_messages(new_messages)
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "qwen3" in self.model_name.lower():
            template_kwargs["enable_thinking"] = False
        text = self.processor.apply_chat_template(
            vision_messages, **template_kwargs)
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

        next_logits = prefill_out.logits[:, -1, :]
        next_token = self._sample(next_logits, temperature)

        # --- Generate response token-by-token ---
        generated_ids = []
        eos_ids = self._get_eos_token_ids()

        for _ in range(max_new_tokens):
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if tok_id in eos_ids:
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
            next_token = self._sample(step_out.logits[:, -1, :], temperature)

        # --- Close the assistant turn in the KV cache ---
        # The decode loop breaks BEFORE feeding EOS through the model, so
        # the KV cache is missing <|im_end|>\n that closes the assistant turn.
        # Feed those tokens now so the next incremental prefill aligns correctly.
        self._close_assistant_turn_in_cache(next_token.device)

        # n + 1: account for the assistant response we just generated so the
        # next call only prefills truly new user messages.
        self.session.processed_turns = n + 1

        response = self._decode_stripping_think_block(generated_ids)

        print(f"[FI] Generated {len(generated_ids)} tokens. "
              f"Cache size: {self._physical_cache_size()} tokens.", flush=True)
        return response

    def reset(self):
        """Clear session state for a new browser task."""
        self.session = SessionState()
        torch.cuda.empty_cache()
        print("[FI] Session reset.", flush=True)

    def session_status(self) -> dict:
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        return {
            "processed_turns": self.session.processed_turns,
            "virtual_position": self.session.virtual_position,
            "physical_kv_size": self._physical_cache_size(),
            "gpu_memory_allocated_mb": round(mem_mb, 1),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample next token: greedy if temperature<=0, else multinomial."""
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _close_assistant_turn_in_cache(self, device):
        """
        Feed ``<|im_end|>\\n`` into the KV cache to properly close the
        assistant turn.  The decode loop stops on EOS *before* its forward
        pass, so the closing token is missing from the cache.  We also need
        the ``\\n`` that Qwen's chat template places between messages.
        """
        tok = self.processor.tokenizer
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        nl_ids = tok.encode("\n", add_special_tokens=False)
        closing_ids = [im_end_id] + nl_ids

        closing_tensor = torch.tensor([closing_ids], device=device)
        cache_pos = torch.arange(
            self.session.virtual_position,
            self.session.virtual_position + len(closing_ids),
            device=device,
        )
        with torch.no_grad():
            self.model(
                input_ids=closing_tensor,
                past_key_values=self.session.kv_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
        self.session.virtual_position += len(closing_ids)

    def _physical_cache_size(self) -> int:
        if self.session.kv_cache is None:
            return 0
        cache = self.session.kv_cache
        if hasattr(cache, 'key_cache'):
            for k in cache.key_cache:
                if k is not None and k.numel() > 0:
                    return k.shape[-2]
        for layer_obj in getattr(cache, 'layers', []):
            if hasattr(layer_obj, 'keys') and layer_obj.keys is not None:
                return layer_obj.keys.shape[-2]
        return 0

    def _get_eos_token_ids(self) -> set[int]:
        """Return all EOS token IDs from the tokenizer and generation config."""
        ids = set()
        tok = self.processor.tokenizer
        if tok.eos_token_id is not None:
            if isinstance(tok.eos_token_id, (list, tuple)):
                ids.update(tok.eos_token_id)
            else:
                ids.add(tok.eos_token_id)
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(self.model_name)
            if gc.eos_token_id is not None:
                if isinstance(gc.eos_token_id, (list, tuple)):
                    ids.update(gc.eos_token_id)
                else:
                    ids.add(gc.eos_token_id)
        except Exception:
            pass
        return ids

    def _decode_stripping_think_block(self, generated_ids: list[int]) -> str:
        """
        Decode generated token IDs, stripping the <think>...</think> reasoning
        block that Qwen3 models produce.  vLLM does this automatically; we need
        to replicate it for parity.

        Strategy: find the </think> token in generated_ids and only decode
        the tokens after it.
        """
        tok = self.processor.tokenizer
        think_end_id = None
        for candidate in ["</think>"]:
            ids = tok.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                think_end_id = ids[0]
                break

        answer_ids = generated_ids
        if think_end_id is not None and think_end_id in generated_ids:
            idx = generated_ids.index(think_end_id)
            candidate = generated_ids[idx + 1:]
            candidate_text = tok.decode(candidate, skip_special_tokens=True).strip()
            if candidate_text:
                answer_ids = candidate
                print(f"[FI] Stripped thinking block ({idx + 1} tokens)", flush=True)
            else:
                print(f"[FI] Think block covered entire response — keeping full output", flush=True)

        response = tok.decode(answer_ids, skip_special_tokens=True).strip()
        return response

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
                            print(f"  [FI] Image decode error: {e}", flush=True)
            out.append({"role": role, "content": new_content})
        return out
