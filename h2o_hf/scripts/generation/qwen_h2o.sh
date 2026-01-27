#!/bin/bash
# Text generation with H2O for Qwen models (LLM only, not VL)

python -u run_text_generation.py \
    --model_arch qwen \
    --model_name Qwen/Qwen2-7B-Instruct \
    --recent_ratio 0.1 \
    --heavy_ratio 0.1
