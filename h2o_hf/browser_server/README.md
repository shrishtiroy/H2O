# FlashInfer Browser Agent Server

A stateful FastAPI server that replaces vLLM for browser agent workloads.
It keeps the KV cache alive across turns and uses FlashInfer fused attention
kernels for fast prefill and decode.

## Why this exists

vLLM discards the KV cache after every response. browser-use sends the full
conversation history on every turn, so vLLM re-prefills everything from scratch
each time — O(n²) cost as the session grows.

This server:
- Keeps the KV cache on GPU between turns
- On each new turn, only prefills the **new** messages (incremental prefill)
- Uses FlashInfer fused kernels for both prefill and decode (falls back to
  PyTorch SDPA on pre-Turing GPUs like V100)
- GQA handled natively — no `repeat_kv` expansion overhead with FlashInfer
- Strips Qwen3's `<think>...</think>` reasoning block automatically
- Handles multiple EOS tokens from the generation config

## Quick start

```bash
# 1. Stop vLLM if running (frees GPUs)
kill $(pgrep -f "vllm serve")

# 2. Install deps (one-time)
pip install fastapi uvicorn flashinfer-python qwen-vl-utils

# 3. Start the server (same port 8001 as vLLM)
cd /home/shroy/git/H2O/h2o_hf
python browser_server/server.py \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --port 8001

# 4. Run your browser agent unchanged
cd /home/shroy/vllm_work/BrowserUseScript
VLLM_URL=http://0.0.0.0:8001/v1 \
MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct \
    python agent_browse.py
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-VL-4B-Instruct` | Any Qwen2/2.5/3-VL model on HuggingFace |
| `--max_pixels` | `1638400` | Max pixels per screenshot (1280x1280) |
| `--device_map` | `auto` | Device map for multi-GPU model sharding |
| `--port` | `8001` | Same as vLLM default |
| `--host` | `0.0.0.0` | Bind address |

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible (browser-use calls this) |
| `GET  /v1/models` | Returns model list (browser-use checks on startup) |
| `POST /v1/reset` | Manually clear session (new browser task) |
| `GET  /v1/session/status` | KV cache size, turn count, GPU memory |
| `GET  /health` | Health check |

## Monitoring session state

```bash
# Check KV cache size and memory between agent steps
curl http://localhost:8001/v1/session/status
# {
#   "processed_turns": 4,
#   "virtual_position": 8432,
#   "physical_kv_size": 8432,
#   "gpu_memory_allocated_mb": 4821.3
# }

# Start a new task without restarting the server
curl -X POST http://localhost:8001/v1/reset
```

## Architecture

```
browser-use agent
      │
      │  POST /v1/chat/completions (OpenAI format)
      ▼
┌─────────────────────────────────────────────┐
│  server.py (FastAPI)                        │
│  - OpenAI-compatible API                    │
│  - asyncio lock for session serialization   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  inference.py (StatefulFlashInferInference) │
│  - Incremental prefill (new messages only)  │
│  - Token-by-token decode loop               │
│  - Think-block stripping for Qwen3          │
│  - Multi-EOS handling                       │
│  - Session state (KV cache, position)       │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  modify_qwen_flashinfer.py                  │
│  - Replaces HF attention layers at load     │
│  - FlashInfer kernels (sm75+ / Turing+):    │
│      single_prefill_with_kv_cache           │
│      single_decode_with_kv_cache            │
│  - PyTorch SDPA fallback (V100 / sm70):     │
│      F.scaled_dot_product_attention          │
└─────────────────────────────────────────────┘
```

## GPU compatibility

| GPU | Compute | Backend |
|-----|---------|---------|
| V100 | sm70 | PyTorch SDPA (automatic fallback) |
| T4, RTX 2080 | sm75 | FlashInfer fused kernels |
| A100, A6000 | sm80 | FlashInfer fused kernels |
| H100 | sm90 | FlashInfer fused kernels |

The backend is detected automatically at startup. No code changes needed.

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI server with OpenAI-compatible endpoints |
| `inference.py` | Stateful inference with incremental prefill and decode loop |
| `../utils_hh/modify_qwen_flashinfer.py` | FlashInfer/SDPA attention layer replacement |
| `../utils_hh/modify_qwen.py` | Original H2O attention (for eviction experiments) |
