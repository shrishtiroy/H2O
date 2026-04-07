#!/usr/bin/env python
"""
H2O DocVQA Benchmark: Baseline vs H2O on document-image VQA.

DocVQA uses high-resolution scanned document images which produce 800-2000 visual
tokens per image — long enough for H2O eviction to activate without forcing
min_seq_for_eviction=0.

Supports Qwen2-VL, Qwen2.5-VL, and Qwen3-VL (detected automatically from model name).

Usage:
    # Quick test (20 samples, 2 H2O ratios)
    python run_docvqa_h2o_benchmark.py --num_samples 20 --h2o_configs "0.1,0.3"

    # Full benchmark (200 samples, default ratios 0.1-0.5)
    python run_docvqa_h2o_benchmark.py --num_samples 200

    # Qwen3-VL (requires: pip install "transformers>=4.53" in the active env)
    python run_docvqa_h2o_benchmark.py --model_name Qwen/Qwen3-VL-2B-Instruct
"""
import argparse
import os
import json
import time
import unicodedata
import re

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoConfig

from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("qwen_vl_utils not found. Install with: pip install qwen-vl-utils")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model_class(model_name: str):
    """Detect model class from model name or config model_type."""
    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        try:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError(
                "Qwen3-VL not available. Run: pip install 'transformers>=4.53'"
            )
    elif "qwen2.5" in name_lower or "qwen2_5" in name_lower:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise ImportError("Qwen2.5-VL not found in transformers.")
    else:
        # Default: Qwen2-VL
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


def load_model(model_name, enable_h2o=False, heavy_ratio=0.1, recent_ratio=0.1,
               min_seq_for_eviction=100):
    """Load model (Qwen2/2.5/3-VL) with optional H2O conversion."""
    label = f"H2O({heavy_ratio}/{recent_ratio})" if enable_h2o else "Baseline"
    print(f"\nLoading {model_name} [{label}]...", flush=True)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if enable_h2o:
        config.heavy_ratio = heavy_ratio
        config.recent_ratio = recent_ratio
        config.min_seq_for_eviction = min_seq_for_eviction

    ModelClass = _get_model_class(model_name)
    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )

    if enable_h2o:
        model = convert_kvcache_qwen_heavy_recent(model, config)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# ANLS metric (DocVQA official evaluation)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip accents, remove punctuation/extra whitespace."""
    text = unicodedata.normalize("NFD", text.lower().strip())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance."""
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = (prev[j - 1] if s[i - 1] == t[j - 1]
                     else 1 + min(prev[j - 1], prev[j], dp[j - 1]))
    return dp[n]


def _extract_answer(text: str) -> str:
    """
    Strip verbose preambles that VLMs add before the actual answer.
    e.g. "The page number is 2." → "2"
        "Based on the document, the answer is Total Revenue." → "Total Revenue"
    Strategy: take the last meaningful phrase after common preamble patterns,
    then fall back to the whole first line.
    """
    # Common VLM preamble patterns
    for pattern in [
        r"(?:the\s+)?answer\s+(?:is|to\s+(?:this|the)\s+question\s+is)[:\s]+",
        r"based\s+on\s+(?:the\s+)?(?:document|image|text)[,\s]+(?:the\s+)?(?:answer\s+is[:\s]+)?",
        r"according\s+to\s+(?:the\s+)?(?:document|image|text)[,\s]+",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[m.end():].strip().rstrip(".")
            break
    # Take just the first line / first sentence
    for sep in ["\n", ". ", "? ", "! "]:
        if sep in text:
            candidate = text.split(sep)[0].strip()
            if candidate:
                text = candidate
                break
    return text.strip()


def anls_score(prediction: str, ground_truths) -> float:
    """
    Average Normalized Levenshtein Similarity for a single sample.
    Computed on both raw prediction and extracted-answer prediction; best wins.
    ground_truths can be a list of acceptable answers.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    candidates = [prediction, _extract_answer(prediction)]
    best = 0.0
    for pred_text in candidates:
        pred_norm = _normalize_text(pred_text)
        for gt in ground_truths:
            gt_norm = _normalize_text(gt)
            max_len = max(len(pred_norm), len(gt_norm))
            if max_len == 0:
                sim = 1.0
            else:
                dist = _levenshtein(pred_norm, gt_norm)
                sim = 1.0 - dist / max_len
            # Threshold: if similarity < 0.5 it counts as 0 (DocVQA convention)
            best = max(best, sim if sim >= 0.5 else 0.0)
    return best


def exact_match(prediction: str, ground_truths) -> bool:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for pred_text in [prediction, _extract_answer(prediction)]:
        pred_norm = _normalize_text(pred_text)
        if any(pred_norm == _normalize_text(gt) for gt in ground_truths):
            return True
    return False


def substring_match(prediction: str, ground_truths) -> bool:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for pred_text in [prediction, _extract_answer(prediction)]:
        pred_norm = _normalize_text(pred_text)
        if any(_normalize_text(gt) in pred_norm or pred_norm in _normalize_text(gt)
               for gt in ground_truths):
            return True
    return False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_docvqa(num_samples: int):
    """Load DocVQA samples. Returns list of dicts with 'image', 'question', 'answers'."""
    print("Loading DocVQA dataset...", flush=True)

    ds = None
    # (dataset_id, config_name, split, img_col, q_col, a_col)
    # trust_remote_code is removed — newer datasets API doesn't support it
    candidates = [
        ("pixparse/docvqa-single-page-questions", None,       "validation", "image", "question", "answers"),
        ("lmms-lab/DocVQA",                       "DocVQA",   "validation", "image", "question", "answers"),
        ("lmms-lab/DocVQA",                       "InfographicVQA", "validation", "image", "question", "answers"),
    ]

    img_col = q_col = a_col = None
    for dataset_id, cfg, split, ic, qc, ac in candidates:
        try:
            if cfg:
                ds = load_dataset(dataset_id, cfg, split=split, streaming=True)
            else:
                ds = load_dataset(dataset_id, split=split, streaming=True)
            # Verify first row is accessible
            first = next(iter(ds))
            img_col, q_col, a_col = ic, qc, ac
            print(f"  Loaded from: {dataset_id}"
                  + (f" (config={cfg})" if cfg else ""), flush=True)
            break
        except Exception as e:
            print(f"  Could not load {dataset_id}: {str(e)[:100]}", flush=True)
            ds = None
            continue

    if ds is None:
        raise RuntimeError(
            "Could not load DocVQA from any source. "
            "Ensure you are logged in: huggingface-cli login"
        )

    samples = []
    for i, row in enumerate(ds):
        if len(samples) >= num_samples:
            break
        img = row.get(img_col)
        q = row.get(q_col, "")
        a = row.get(a_col, [])
        if img is None or not q:
            continue
        # Normalise answers to list
        if isinstance(a, str):
            a = [a]
        samples.append({"image": img, "question": q, "answers": a})

    print(f"  Using {len(samples)} samples", flush=True)
    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def reset_h2o_state(model):
    """Reset H2O attention state across all layers."""
    for module in model.modules():
        if hasattr(module, '_reset_masks'):
            module._reset_masks()
        if hasattr(module, 'h2o_scores'):
            module.h2o_scores = None
            module.attention_masks_next = None
            module.heavy_budget = None
            module.recent_budget = None
            module.cache_budget = None


def run_inference(model, processor, sample, max_pixels=1280 * 1280):
    """Run single VQA inference, return (prediction, num_tokens)."""
    reset_h2o_state(model)

    image = sample["image"]
    question = sample["question"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "max_pixels": max_pixels},
            {"type": "text",  "text": question},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    num_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    prediction = processor.decode(out[0][num_tokens:], skip_special_tokens=True).strip()
    return prediction, num_tokens


def evaluate(model, processor, samples, label=""):
    """Evaluate model on all samples, return metrics dict."""
    total_anls = 0.0
    total_em = 0
    total_sub = 0
    total_tokens = []

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for i, sample in enumerate(samples):
        try:
            pred, num_tok = run_inference(model, processor, sample)
        except Exception as e:
            print(f"  [{label}] Sample {i}: ERROR — {e}", flush=True)
            pred, num_tok = "", 0

        total_tokens.append(num_tok)
        total_anls += anls_score(pred, sample["answers"])
        total_em   += int(exact_match(pred, sample["answers"]))
        total_sub  += int(substring_match(pred, sample["answers"]))

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            anls_so_far = total_anls / (i + 1)
            print(f"  [{label}] {i+1}/{len(samples)} | "
                  f"ANLS={anls_so_far:.3f} | tokens={num_tok}", flush=True)

    elapsed = time.time() - t0
    n = len(samples)
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "label": label,
        "num_samples": n,
        "anls": round(total_anls / n, 4),
        "exact_match_pct": round(100 * total_em / n, 1),
        "substring_match_pct": round(100 * total_sub / n, 1),
        "avg_input_tokens": round(sum(total_tokens) / n, 1),
        "peak_memory_mb": round(peak_mem, 1),
        "time_seconds": round(elapsed, 1),
        "throughput_samples_per_sec": round(n / elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="H2O DocVQA Benchmark: Baseline vs H2O on document-image VQA")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--h2o_configs", type=str, default="0.1,0.2,0.3",
                        help="Comma-separated ratios (used for both heavy and recent). "
                             "cache_budget = (ratio+ratio)*prefill_len, so ratio=0.5 means "
                             "budget=prefill_len and nothing gets evicted. Use 0.1-0.3.")
    parser.add_argument("--min_seq_for_eviction", type=int, default=100,
                        help="H2O activates only when KV length >= this. "
                             "Set low (e.g. 100) to ensure activation on DocVQA sequences.")
    parser.add_argument("--max_pixels", type=int, default=600 * 800,
                        help="Max pixels for image tiling. Default 600x800=480000 keeps "
                             "sequences at ~800-1400 tokens, well within the 2048 context "
                             "limit and giving H2O room to operate. 1280*1280 causes "
                             "sequences to hit the 2048 cap.")
    parser.add_argument("--output_path", type=str,
                        default="./benchmark_results/docvqa_h2o_results_0.4.json")
    args = parser.parse_args()

    ratios = [float(r) for r in args.h2o_configs.split(",")]

    print(f"Model:  {args.model_name}")
    print(f"H2O configs: {ratios}")
    print(f"Samples: {args.num_samples}")
    print(f"min_seq_for_eviction: {args.min_seq_for_eviction}")
    print(f"max_pixels: {args.max_pixels:,}", flush=True)

    # Load dataset once
    samples = load_docvqa(args.num_samples)

    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True)

    all_results = []

    # --- Baseline ---
    model = load_model(args.model_name, enable_h2o=False)
    baseline = evaluate(model, processor, samples, label="Baseline")
    all_results.append(baseline)
    del model
    torch.cuda.empty_cache()

    # --- H2O at each ratio ---
    for ratio in ratios:
        model = load_model(
            args.model_name, enable_h2o=True,
            heavy_ratio=ratio, recent_ratio=ratio,
            min_seq_for_eviction=args.min_seq_for_eviction,
        )
        label = f"H2O({ratio}/{ratio})"
        result = evaluate(model, processor, samples, label=label)
        result["heavy_ratio"] = ratio
        result["recent_ratio"] = ratio
        all_results.append(result)
        del model
        torch.cuda.empty_cache()

    # --- Print table ---
    baseline_anls = all_results[0]["anls"]
    print(f"\n{'='*90}")
    print(f"RESULTS: {args.model_name} — DocVQA ({len(samples)} samples)")
    print(f"{'='*90}")
    print(f"{'Config':<22} {'ANLS':>6} {'dANLS':>7} "
          f"{'EM%':>6} {'Sub%':>6} {'Mem(MB)':>9} "
          f"{'Samp/s':>7} {'AvgTok':>8}")
    print(f"{'-'*80}")

    for r in all_results:
        d_anls = r["anls"] - baseline_anls
        d_str = f"+{d_anls:.3f}" if d_anls >= 0 else f"{d_anls:.3f}"
        print(f"{r['label']:<22} {r['anls']:>6.3f} {d_str:>7} "
              f"{r['exact_match_pct']:>6.1f} {r['substring_match_pct']:>6.1f} "
              f"{r['peak_memory_mb']:>9.0f} "
              f"{r['throughput_samples_per_sec']:>7.2f} "
              f"{r['avg_input_tokens']:>8.0f}")

    print(f"{'='*90}")

    # --- Save JSON ---
    output = {
        "model": args.model_name,
        "dataset": "DocVQA",
        "num_samples": len(samples),
        "min_seq_for_eviction": args.min_seq_for_eviction,
        "max_pixels": args.max_pixels,
        "results": all_results,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
