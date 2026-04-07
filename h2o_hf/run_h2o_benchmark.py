#!/usr/bin/env python
"""
H2O KV Cache Benchmark: Perplexity on WikiText-2.

Measures perplexity, peak GPU memory, and throughput for baseline (full KV cache)
vs H2O at multiple heavy/recent ratios on long text sequences.

Usage:
    # Default: TinyLlama, 10 samples of 2048 tokens, H2O ratios 0.1-0.5
    python run_h2o_benchmark.py

    # Quick test
    python run_h2o_benchmark.py --num_samples 2 --seq_length 1024

    # Custom model and ratios
    python run_h2o_benchmark.py --model_name meta-llama/Llama-2-7b-hf --h2o_configs 0.1,0.3,0.5
"""
import argparse
import os
import sys
import time
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent


def reset_h2o_state(model):
    """Reset H2O attention state in all layers."""
    for module in model.modules():
        if hasattr(module, '_reset_masks'):
            module._reset_masks()


def get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def prepare_data(tokenizer, seq_length=2048, num_samples=10):
    """Load WikiText-2 test set, tokenize, split into chunks."""
    print(f"Loading WikiText-2 test set...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    full_text = "\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    all_ids = encodings.input_ids[0]

    chunks = []
    for i in range(0, len(all_ids) - seq_length + 1, seq_length):
        chunk = all_ids[i : i + seq_length].unsqueeze(0)
        chunks.append(chunk)
        if len(chunks) >= num_samples:
            break

    print(f"Prepared {len(chunks)} chunks of {seq_length} tokens "
          f"(from {len(all_ids)} total tokens)", flush=True)
    return chunks


def load_model(model_name, enable_h2o=False, heavy_ratio=0.1, recent_ratio=0.1):
    """Load model with optional H2O conversion."""
    label = f"H2O({heavy_ratio}/{recent_ratio})" if enable_h2o else "Baseline"
    print(f"\nLoading {model_name} [{label}]...", flush=True)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if enable_h2o:
        config.heavy_ratio = heavy_ratio
        config.recent_ratio = recent_ratio

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if enable_h2o:
        model = convert_kvcache_llama_heavy_recent(model, config)

    model.half().to("cuda").eval()
    return model


def compute_perplexity_baseline(model, chunks, device, prefill_ratio=0.5):
    """
    Prefill-then-decode perplexity for baseline (full KV cache).

    Uses the same prefill-then-decode pattern as H2O evaluation for a
    fair comparison. NLL is measured on the same decoded portion.
    """
    total_nll = 0.0
    total_tokens = 0

    for chunk_idx, input_ids in enumerate(chunks):
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]
        prefill_len = max(1, int(seq_len * prefill_ratio))

        chunk_nll = 0.0
        chunk_tokens = 0

        # Phase 1: Prefill
        with torch.no_grad():
            outputs = model(input_ids[:, :prefill_len], use_cache=True)
        past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        target = input_ids[:, prefill_len]
        loss = F.cross_entropy(logits.float(), target)
        chunk_nll += loss.item()
        chunk_tokens += 1

        # Phase 2: Decode
        for i in range(prefill_len, seq_len - 1):
            token = input_ids[:, i:i+1]
            with torch.no_grad():
                outputs = model(
                    token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            target = input_ids[:, i + 1]
            loss = F.cross_entropy(logits.float(), target)
            chunk_nll += loss.item()
            chunk_tokens += 1

        total_nll += chunk_nll
        total_tokens += chunk_tokens
        chunk_ppl = torch.exp(torch.tensor(chunk_nll / chunk_tokens)).item()

        if (chunk_idx + 1) % 5 == 0 or chunk_idx == len(chunks) - 1:
            ppl_so_far = torch.exp(torch.tensor(total_nll / total_tokens)).item()
            print(f"  Baseline: {chunk_idx+1}/{len(chunks)} chunks, PPL so far: {ppl_so_far:.2f}", flush=True)

    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity, avg_nll


def set_h2o_budgets(model, seq_length, heavy_ratio, recent_ratio):
    """Pre-set H2O budgets based on full sequence length.

    When processing token-by-token, the budget is normally computed from
    the first token's input_len (=1), giving a budget of 1 token. This
    pre-sets the budgets to the correct values based on the full sequence.
    """
    for module in model.modules():
        if hasattr(module, 'heavy_budget_ratio'):
            module.heavy_budget = max(1, int(heavy_ratio * seq_length))
            module.recent_budget = max(1, int(recent_ratio * seq_length))
            module.cache_budget = module.heavy_budget + module.recent_budget


def compute_perplexity_h2o(model, chunks, device, heavy_ratio, recent_ratio,
                           prefill_ratio=0.5):
    """
    Prefill-then-decode perplexity for H2O evaluation.

    Mirrors real usage: process a prefix in one forward pass (prefill),
    then decode remaining tokens one at a time. H2O eviction activates
    once KV cache exceeds the budget during the decode phase.

    NLL is measured only on the decoded tokens (after prefill).
    """
    total_nll = 0.0
    total_tokens = 0

    for chunk_idx, input_ids in enumerate(chunks):
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]
        prefill_len = max(1, int(seq_len * prefill_ratio))

        # Reset H2O state and pre-set budgets based on prefill length
        reset_h2o_state(model)
        set_h2o_budgets(model, prefill_len, heavy_ratio, recent_ratio)

        chunk_nll = 0.0
        chunk_tokens = 0

        # Phase 1: Prefill - process prefix in one forward pass
        with torch.no_grad():
            outputs = model(
                input_ids[:, :prefill_len],
                use_cache=True,
            )
        past_key_values = outputs.past_key_values

        # Get NLL for the last prefill token predicting first decode token
        logits = outputs.logits[:, -1, :]
        target = input_ids[:, prefill_len]
        loss = F.cross_entropy(logits.float(), target)
        chunk_nll += loss.item()
        chunk_tokens += 1

        # Phase 2: Decode - process remaining tokens one at a time
        for i in range(prefill_len, seq_len - 1):
            token = input_ids[:, i:i+1]

            with torch.no_grad():
                outputs = model(
                    token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            target = input_ids[:, i + 1]

            loss = F.cross_entropy(logits.float(), target)
            chunk_nll += loss.item()
            chunk_tokens += 1

        total_nll += chunk_nll
        total_tokens += chunk_tokens
        chunk_ppl = torch.exp(torch.tensor(chunk_nll / chunk_tokens)).item()
        print(f"  H2O: chunk {chunk_idx+1}/{len(chunks)}, "
              f"chunk PPL: {chunk_ppl:.2f} "
              f"(prefill={prefill_len}, decode={seq_len - prefill_len})",
              flush=True)

    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity, avg_nll


def evaluate_config(model_name, chunks, heavy_ratio=None, recent_ratio=None):
    """Evaluate a single configuration and return metrics dict."""
    is_h2o = heavy_ratio is not None
    label = f"H2O({heavy_ratio}/{recent_ratio})" if is_h2o else "Baseline"

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}", flush=True)

    model = load_model(
        model_name,
        enable_h2o=is_h2o,
        heavy_ratio=heavy_ratio or 0.0,
        recent_ratio=recent_ratio or 0.0,
    )
    device = next(model.parameters()).device

    torch.cuda.empty_cache()
    reset_peak_memory()

    start_time = time.time()

    if is_h2o:
        perplexity, avg_nll = compute_perplexity_h2o(
            model, chunks, device, heavy_ratio, recent_ratio)
    else:
        perplexity, avg_nll = compute_perplexity_baseline(model, chunks, device)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()
    total_tokens = sum(c.shape[1] for c in chunks)
    throughput = total_tokens / elapsed

    del model
    torch.cuda.empty_cache()

    result = {
        "label": label,
        "heavy_ratio": heavy_ratio,
        "recent_ratio": recent_ratio,
        "perplexity": round(perplexity, 2),
        "avg_nll": round(avg_nll, 4),
        "peak_memory_mb": round(peak_mem, 1),
        "time_seconds": round(elapsed, 1),
        "throughput_tok_per_sec": round(throughput, 1),
        "total_tokens": total_tokens,
    }

    print(f"\n  Perplexity: {perplexity:.2f}")
    print(f"  Peak Memory: {peak_mem:.1f} MB")
    print(f"  Throughput: {throughput:.1f} tokens/sec")
    print(f"  Time: {elapsed:.1f}s", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="H2O KV Cache Benchmark: Perplexity on WikiText-2")
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--h2o_configs", type=str, default="0.1,0.2,0.3,0.5",
                        help="Comma-separated ratios (used for both heavy and recent)")
    parser.add_argument("--output_path", type=str,
                        default="./benchmark_results/h2o_benchmark_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ratios = [float(r) for r in args.h2o_configs.split(",")]

    print(f"Model: {args.model_name}")
    print(f"Seq length: {args.seq_length}, Samples: {args.num_samples}")
    print(f"H2O configs: {ratios}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = prepare_data(tokenizer, args.seq_length, args.num_samples)

    if len(chunks) < args.num_samples:
        print(f"WARNING: Only {len(chunks)} chunks available "
              f"(requested {args.num_samples})")

    all_results = []

    # Baseline
    all_results.append(evaluate_config(args.model_name, chunks))

    # H2O at each ratio
    for ratio in ratios:
        all_results.append(
            evaluate_config(args.model_name, chunks,
                            heavy_ratio=ratio, recent_ratio=ratio))

    # Print results table
    baseline_ppl = all_results[0]["perplexity"]

    print(f"\n{'='*80}")
    print(f"RESULTS: {args.model_name}")
    print(f"WikiText-2, seq_length={args.seq_length}, num_samples={len(chunks)}")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'PPL':>8} {'dPPL':>8} "
          f"{'Memory(MB)':>12} {'Tok/s':>10} {'Time(s)':>10}")
    print(f"{'-'*70}")

    for r in all_results:
        delta = r["perplexity"] - baseline_ppl
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        print(f"{r['label']:<20} {r['perplexity']:>8.2f} "
              f"{delta_str:>8} {r['peak_memory_mb']:>12.1f} "
              f"{r['throughput_tok_per_sec']:>10.1f} "
              f"{r['time_seconds']:>10.1f}")

    print(f"{'='*80}")

    # Save JSON
    output = {
        "model": args.model_name,
        "dataset": "wikitext-2-raw-v1",
        "seq_length": args.seq_length,
        "num_samples": len(chunks),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
