#!/bin/bash
# VQA evaluation with H2O for Qwen2-VL / Qwen3-VL

# Sanity check (quick test)
python -u run_vqa_eval.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --sanity_check \
    --enable_h2o \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1

# Full VQA evaluation (uncomment to run)
# python -u run_vqa_eval.py \
#     --model_name Qwen/Qwen2-VL-7B-Instruct \
#     --tasks vqav2_val_lite \
#     --enable_h2o \
#     --heavy_ratio 0.1 \
#     --recent_ratio 0.1 \
#     --output_path ./vqa_results
