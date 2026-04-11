"""
SGLangH2OInference -- drop-in replacement for StatefulH2OInference that uses
SGLang's serving infrastructure (FlashInfer attention, PagedAttention KV
cache, RadixAttention prefix caching).

Session persistence across browser-agent turns is handled by RadixAttention:
when the agent sends the full conversation each turn, SGLang reuses the
cached KV for the shared prefix and only computes new tokens.

When the context exceeds the model's window, old messages are pruned at the
message level (whole messages removed, oldest first, system prompt kept).

Usage:
    inf = SGLangH2OInference(model_name="Qwen/Qwen3-VL-2B-Instruct",
                              heavy_ratio=0.1, recent_ratio=0.1)
    response = inf.chat(messages)   # same API as StatefulH2OInference
"""

import base64
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install qwen-vl-utils:  pip install qwen-vl-utils")


@dataclass
class SGLangSessionState:
    processed_turns: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0


class SGLangH2OInference:
    """
    Wraps an SGLang Engine for Qwen-VL browser agent serving.

    Matches the public API of ``StatefulH2OInference`` so that
    ``server.py`` can swap backends transparently.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        heavy_ratio: float = 0.1,
        recent_ratio: float = 0.1,
        min_seq_for_eviction: int = 500,
        max_pixels: int = 1280 * 1280,
        device_map: str = "auto",
        tp_size: int = 1,
        attention_backend: str = "flashinfer",
    ):
        self.model_name = model_name
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.max_pixels = max_pixels

        print(f"[SGLang H2O] Loading {model_name} ...", flush=True)

        from transformers import AutoConfig, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        text_cfg = getattr(config, "text_config", config)
        self.max_context_length = getattr(
            text_cfg, "max_position_embeddings", 32768
        )
        self.context_budget = int(
            self.max_context_length * (heavy_ratio + recent_ratio)
        )

        # ---- Optionally enable H2O custom attention backend ----
        use_h2o_backend = attention_backend == "h2o_flashinfer"
        if use_h2o_backend:
            self._setup_h2o_backend(min_seq_for_eviction)
            backend_name = "h2o_flashinfer"
        else:
            backend_name = "flashinfer"

        # ---- Start SGLang engine ----
        import sglang

        engine_kwargs = dict(
            model_path=model_name,
            attention_backend=backend_name,
            tp_size=tp_size,
        )

        self.engine = sglang.Engine(**engine_kwargs)
        self.session = SGLangSessionState()

        print(
            f"[SGLang H2O] Ready.  backend={backend_name}  "
            f"context_budget={self.context_budget}  "
            f"max_ctx={self.max_context_length}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    #  Public API (same as StatefulH2OInference)                          #
    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        n = len(messages)

        if n < self.session.processed_turns:
            print(
                f"[SGLang H2O] New session detected "
                f"(msg count {n} < {self.session.processed_turns}). Resetting.",
                flush=True,
            )
            self.reset()

        new_count = n - self.session.processed_turns
        print(
            f"[SGLang H2O] Turn {self.session.processed_turns // 2 + 1}: "
            f"processing {new_count} new message(s)",
            flush=True,
        )

        vision_msgs = self._build_vision_messages(messages)

        text = self.processor.apply_chat_template(
            vision_msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(vision_msgs)

        t0 = time.time()
        result = self.engine.generate(
            prompt=text,
            image_data=image_inputs if image_inputs else None,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        )
        elapsed = time.time() - t0

        response_text = result["text"]
        meta = result.get("meta_info", {})
        gen_ids = meta.get("completion_tokens", 0)
        prompt_toks = meta.get("prompt_tokens", 0)

        self.session.processed_turns = n
        self.session.total_tokens_generated += gen_ids
        self.session.total_prompt_tokens = prompt_toks

        print(
            f"[SGLang H2O] Generated {gen_ids} tokens in {elapsed:.1f}s  "
            f"(prompt={prompt_toks})",
            flush=True,
        )
        return response_text

    def reset(self):
        self.session = SGLangSessionState()
        try:
            self.engine.flush_cache()
        except Exception:
            pass
        print("[SGLang H2O] Session reset.", flush=True)

    def session_status(self) -> dict:
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        return {
            "processed_turns": self.session.processed_turns,
            "total_tokens_generated": self.session.total_tokens_generated,
            "total_prompt_tokens": self.session.total_prompt_tokens,
            "context_budget": self.context_budget,
            "gpu_memory_allocated_mb": round(mem_mb, 1),
            "backend": "sglang",
        }

    # ------------------------------------------------------------------ #
    #  H2O backend setup                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _setup_h2o_backend(min_seq_for_eviction: int):
        """
        Patch SGLang's attention registry so that spawned scheduler
        subprocesses can discover the ``h2o_flashinfer`` backend.
        """
        h2o_mod = "browser_server.sglang_h2o_backend"

        h2o_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        if h2o_root not in sys.path:
            sys.path.insert(0, h2o_root)

        from browser_server.sglang_h2o_backend import _patch_attention_registry

        ok = _patch_attention_registry()
        if ok:
            os.environ["SGLANG_H2O_BACKEND_MODULE"] = h2o_mod
            os.environ.setdefault(
                "SGLANG_H2O_MIN_SEQ", str(min_seq_for_eviction)
            )
            print("[SGLang H2O] Attention registry patched for h2o_flashinfer.")
        else:
            print(
                "[SGLang H2O] WARNING: could not patch attention_registry.py. "
                "Falling back to standard flashinfer.",
            )

    # ------------------------------------------------------------------ #
    #  Message building / image handling                                  #
    # ------------------------------------------------------------------ #

    def _build_vision_messages(self, messages: list[dict]) -> list[dict]:
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
                    new_content.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url:
                        try:
                            img = _decode_image(url)
                            new_content.append(
                                {
                                    "type": "image",
                                    "image": img,
                                    "max_pixels": self.max_pixels,
                                }
                            )
                        except Exception as e:
                            print(f"  [SGLang H2O] Image decode error: {e}")
            out.append({"role": role, "content": new_content})
        return out


# ---------------------------------------------------------------------- #
#  Helpers                                                                #
# ---------------------------------------------------------------------- #

def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
    else:
        b64 = url
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGB")
