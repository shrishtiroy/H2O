# H2O Browser Agent Server

A stateful FastAPI server that replaces vLLM for browser agent workloads.
It keeps the KV cache alive across turns and uses H2O eviction to keep memory bounded.

## Why this exists

vLLM discards the KV cache after every response.  Browser-use sends the full
conversation history on every turn, so vLLM re-prefills everything from scratch
each time — O(n²) cost as the session grows.

This server:
- Keeps the KV cache on GPU between turns
- On each new turn, only prefills the **new** messages
- After each turn, physically removes evicted KV rows (hard eviction)
- Memory stays bounded at `(heavy_ratio + recent_ratio) × prefill_len`

## Usage

```bash
# 1. Stop vLLM (frees all GPUs)
kill $(pgrep -f "vllm serve")

# 2. Install deps (one-time)
conda activate h2o
pip install fastapi uvicorn

# 3. Start the H2O server (same port 8001 as vLLM)
cd /home/shroy/git/H2O/h2o_hf
python browser_server/server.py \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1 \
    --port 8001

# 4. Run your browser agent unchanged
cd /home/shroy/vllm_work/BrowserUseScript
VLLM_URL=http://0.0.0.0:8001/v1 \
MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
    python agent_browse.py
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | Qwen/Qwen3-VL-2B-Instruct | Any Qwen2/2.5/3-VL model |
| `--heavy_ratio` | 0.1 | Keep top-10% attention tokens |
| `--recent_ratio` | 0.1 | Always keep most recent 10% |
| `--min_seq_for_eviction` | 500 | H2O only fires past this token count |
| `--max_pixels` | 1638400 | Max pixels per screenshot (1280×1280) |
| `--port` | 8001 | Same as vLLM default |

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
#   "physical_kv_size": 312,    ← stays bounded (not 8432) after eviction
#   "heavy_budget": 150,
#   "recent_budget": 150,
#   "gpu_memory_allocated_mb": 4821.3
# }

# Start a new task without restarting the server
curl -X POST http://localhost:8001/v1/reset
```

## How hard eviction works

After every generated response:
1. Read `attention_masks_next` from each H2O attention layer (which tokens survived)
2. Take the union across all heads and all layers (keep a position if **any** layer wants it)
3. Index-select only the surviving rows from the KV cache tensors
4. Prune the accumulated H2O scores to match

The `virtual_position` counter is **not** reset after eviction — RoPE encodings are
already baked into the stored key tensors, so position continuity for new tokens is
maintained correctly even though old tokens have been removed.
