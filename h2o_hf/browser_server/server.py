"""
FlashInfer Browser Agent Server — drop-in replacement for the vLLM server.

Serves an OpenAI-compatible /v1/chat/completions endpoint on port 8001
(same port as the existing vLLM setup).  browser-use scripts need zero changes —
just point VLLM_URL at this server instead of vLLM.

Key features:
  - KV cache is kept alive across turns (stateful session)
  - FlashInfer fused kernels for both prefill and decode attention
  - GQA handled natively by FlashInfer (no repeat_kv overhead)
  - New turns only prefill NEW messages, not the full history

Usage:
    cd /home/shroy/git/H2O/h2o_hf
    python browser_server/server.py \\
        --model_name Qwen/Qwen3-VL-4B-Instruct \\
        --port 8001

    # Run browser agent unchanged
    VLLM_URL=http://0.0.0.0:8001/v1 MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct \\
        python /home/shroy/vllm_work/BrowserUseScript/agent_browse.py
"""

import argparse
import asyncio
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import StatefulFlashInferInference


# ---------------------------------------------------------------------------
# Request / response schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, list[ContentPart]]

    def to_dict(self) -> dict:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        parts = []
        for p in self.content:
            if p.type == "text":
                parts.append({"type": "text", "text": p.text or ""})
            elif p.type == "image_url" and p.image_url:
                parts.append({"type": "image_url",
                               "image_url": {"url": p.image_url.url}})
        return {"role": self.role, "content": parts}

class ChatCompletionRequest(BaseModel):
    model: str = "flashinfer"
    messages: list[Message]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="FlashInfer Browser Server")
inference = None
session_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    """browser-use checks this endpoint on startup."""
    return {
        "object": "list",
        "data": [{
            "id": inference.model_name if inference else "flashinfer",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "flashinfer",
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Maintains stateful KV cache across successive browser-agent turns.
    """
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if request.stream:
        pass

    messages_dicts = [m.to_dict() for m in request.messages]
    max_new_tokens = (request.max_completion_tokens
                      or request.max_tokens
                      or 4096)

    async with session_lock:
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None,
            inference.chat,
            messages_dicts,
            max_new_tokens,
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = getattr(inference.session, "virtual_position", 0)
    completion_tokens = len(response_text.split())

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.post("/v1/reset")
async def reset_session():
    """Clear the KV cache and start a fresh browser task."""
    async with session_lock:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, inference.reset)
    return {"status": "ok", "message": "Session reset"}


@app.get("/v1/session/status")
async def session_status():
    """Inspect current KV cache size, turn count, and GPU memory."""
    if inference is None:
        return {"error": "not loaded"}
    return inference.session_status()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashInfer Stateful Browser Agent Server (drop-in vLLM replacement)")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-4B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--max_pixels", type=int, default=1280 * 1280,
                        help="Max pixels for image tiling (browser screenshots)")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model loading (default: auto)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model with 4-bit quantization (fits 7-8B on 16GB)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port (default 8001 matches existing vLLM setup)")
    args = parser.parse_args()

    global inference
    inference = StatefulFlashInferInference(
        model_name=args.model_name,
        max_pixels=args.max_pixels,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"\n[FlashInfer Server] Backend: FlashInfer (fused attention)", flush=True)
    print(f"[FlashInfer Server] Listening on {args.host}:{args.port}", flush=True)
    print(f"[FlashInfer Server] Drop-in replacement for vLLM — same port, same API",
          flush=True)
    print(f"[FlashInfer Server] Set VLLM_URL=http://{args.host}:{args.port}/v1 "
          f"in your agent\n", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
