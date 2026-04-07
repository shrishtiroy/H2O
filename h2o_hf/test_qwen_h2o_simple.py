#!/usr/bin/env python
"""
Simple side-by-side test: Qwen2-VL baseline vs H2O on a single image.

Forces H2O to activate by setting min_seq_for_eviction=0 so eviction
kicks in even on short (~420 token) VQA sequences.
"""
import torch
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

QUESTIONS = [
    "Describe this image in detail.",
    "What colors do you see in this image?",
    "What objects are in this image?",
]


def load_baseline():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    return model


def load_h2o(heavy_ratio=0.1, recent_ratio=0.1):
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config.heavy_ratio = heavy_ratio
    config.recent_ratio = recent_ratio
    config.min_seq_for_eviction = 0  # Force H2O on short sequences

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda",
    )
    checkpoint = model.state_dict()
    model = convert_kvcache_qwen_heavy_recent(model, config)
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    model.eval()
    return model


def ask(model, processor, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_URL},
        {"type": "text", "text": question},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    num_tokens = inputs['input_ids'].shape[1]

    # Reset H2O state
    for m in model.modules():
        if hasattr(m, '_reset_masks'):
            m._reset_masks()
        if hasattr(m, 'h2o_scores'):
            m.h2o_scores = None
            m.attention_masks_next = None
            m.heavy_budget = None
            m.recent_budget = None
            m.cache_budget = None

    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    elapsed = time.time() - start

    response = processor.decode(out[0][num_tokens:], skip_special_tokens=True)
    return response, num_tokens, elapsed


def main():
    processor = AutoProcessor.from_pretrained(MODEL)
    baseline_results = {}
    h2o_results = {}

    # --- Run baseline ---
    print("=" * 70)
    print("Loading Baseline model...")
    print("=" * 70)
    model_b = load_baseline()
    for q in QUESTIONS:
        resp, tokens, elapsed = ask(model_b, processor, q)
        baseline_results[q] = (resp, tokens, elapsed)
    del model_b
    torch.cuda.empty_cache()

    # --- Run H2O ---
    print("\n" + "=" * 70)
    print("Loading H2O model (0.1/0.1 = 20% cache retention)...")
    print("=" * 70)
    model_h = load_h2o(heavy_ratio=0.1, recent_ratio=0.1)
    for q in QUESTIONS:
        resp, tokens, elapsed = ask(model_h, processor, q)
        h2o_results[q] = (resp, tokens, elapsed)
    del model_h
    torch.cuda.empty_cache()

    # --- Print comparison ---
    for q in QUESTIONS:
        resp_b, tokens_b, time_b = baseline_results[q]
        resp_h, tokens_h, time_h = h2o_results[q]

        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print(f"{'='*70}")
        print(f"\n[Baseline] ({tokens_b} input tokens, {time_b:.1f}s)")
        print(resp_b)
        print(f"\n[H2O 0.1/0.1] ({tokens_h} input tokens, {time_h:.1f}s)")
        print(resp_h)

        if resp_b.strip() == resp_h.strip():
            print("\n>> SAME OUTPUT")
        else:
            print("\n>> DIFFERENT OUTPUT")


if __name__ == "__main__":
    main()
