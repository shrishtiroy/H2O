#!/usr/bin/env python
"""
Long-Context QA Benchmark for H2O KV Cache Evaluation.

Constructs long-context QA prompts from SQuAD by embedding the answer passage
among distractor passages, pushing total input length past the H2O eviction
threshold. Measures accuracy, F1, perplexity, memory, and throughput.

Usage:
    # Default: TinyLlama, 50 samples, ~1800 token contexts
    python run_longcontext_qa_benchmark.py

    # Quick test
    python run_longcontext_qa_benchmark.py --num_samples 5

    # Custom ratios
    python run_longcontext_qa_benchmark.py --h2o_configs 0.1,0.2,0.5
"""
import argparse
import collections
import json
import os
import re
import string
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent


# ============================================================
# Utilities
# ============================================================

def reset_h2o_state(model):
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


# ============================================================
# Answer evaluation (SQuAD-style)
# ============================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_substring_match(prediction, ground_truth):
    """Check if the ground truth appears as a substring in the prediction."""
    return int(normalize_answer(ground_truth) in normalize_answer(prediction))


def score_prediction(prediction, ground_truths):
    """Score a prediction against multiple ground truth answers."""
    em = max(compute_exact_match(prediction, gt) for gt in ground_truths)
    f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    substring = max(compute_substring_match(prediction, gt) for gt in ground_truths)
    return {"exact_match": em, "f1": f1, "substring_match": substring}


# ============================================================
# Data preparation
# ============================================================

def prepare_longcontext_qa(tokenizer, num_samples=50, target_context_tokens=1800,
                           seed=42):
    """Build long-context QA samples from SQuAD.

    For each sample:
    1. Pick a QA pair with its context passage
    2. Pad with distractor passages from other questions until we hit
       target_context_tokens
    3. Place the gold passage at a random position (beginning, middle, end)
    4. Format as a chat prompt asking the model to answer from the context

    Returns list of dicts with keys:
        prompt_ids, prompt_text, question, answers, gold_passage_position,
        num_context_tokens, num_passages
    """
    import random
    random.seed(seed)

    print("Loading SQuAD dataset...", flush=True)
    squad = load_dataset("rajpurkar/squad", split="validation")

    # Group by context to avoid duplicate passages
    context_to_qas = {}
    for item in squad:
        ctx = item['context']
        if ctx not in context_to_qas:
            context_to_qas[ctx] = []
        context_to_qas[ctx].append({
            'question': item['question'],
            'answers': item['answers']['text'],
        })

    contexts = list(context_to_qas.keys())
    random.shuffle(contexts)

    samples = []
    used_contexts = set()

    for gold_ctx in contexts:
        if len(samples) >= num_samples:
            break

        qas = context_to_qas[gold_ctx]
        qa = qas[0]  # Take first QA pair for this context

        if not qa['answers']:
            continue

        # Collect distractor passages
        distractors = []
        for ctx in contexts:
            if ctx == gold_ctx or ctx in used_contexts:
                continue
            distractors.append(ctx)
            if len(distractors) >= 30:
                break

        if len(distractors) < 3:
            continue

        # Build context by adding passages until we hit target length
        # Place gold passage at random position
        passages = []
        current_tokens = 0

        # Tokenize gold passage to know its size
        gold_tokens = len(tokenizer.encode(gold_ctx, add_special_tokens=False))

        # Add distractors until we approach target
        for dist_ctx in distractors:
            dist_tokens = len(tokenizer.encode(dist_ctx, add_special_tokens=False))
            if current_tokens + dist_tokens + gold_tokens > target_context_tokens - 100:
                break
            passages.append(dist_ctx)
            current_tokens += dist_tokens

        if len(passages) < 2:
            continue

        # Insert gold passage at random position
        position_choices = ['beginning', 'middle', 'end']
        position = random.choice(position_choices)
        if position == 'beginning':
            insert_idx = 0
        elif position == 'end':
            insert_idx = len(passages)
        else:
            insert_idx = len(passages) // 2

        passages.insert(insert_idx, gold_ctx)

        # Format the context
        numbered_passages = []
        for i, p in enumerate(passages):
            numbered_passages.append(f"[Passage {i+1}] {p}")
        full_context = "\n\n".join(numbered_passages)

        # Format prompt
        prompt = (
            f"Read the following passages and answer the question based only "
            f"on the information provided.\n\n"
            f"{full_context}\n\n"
            f"Question: {qa['question']}\n"
            f"Answer:"
        )

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        num_tokens = prompt_ids.shape[1]

        # Skip if too short or too long for the model
        if num_tokens < 512 or num_tokens > 2000:
            continue

        used_contexts.add(gold_ctx)
        samples.append({
            'prompt_ids': prompt_ids,
            'prompt_text': prompt,
            'question': qa['question'],
            'answers': qa['answers'],
            'gold_passage_position': position,
            'num_context_tokens': num_tokens,
            'num_passages': len(passages),
        })

    print(f"Prepared {len(samples)} long-context QA samples", flush=True)
    if samples:
        token_counts = [s['num_context_tokens'] for s in samples]
        print(f"  Token range: {min(token_counts)}-{max(token_counts)}, "
              f"avg: {sum(token_counts)/len(token_counts):.0f}", flush=True)
    return samples


# ============================================================
# Model loading
# ============================================================

def load_model(model_name, enable_h2o=False, heavy_ratio=0.1, recent_ratio=0.1):
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


# ============================================================
# Evaluation
# ============================================================

def evaluate_qa(model, tokenizer, samples, device, max_new_tokens=50):
    """Run QA evaluation. Returns metrics dict and per-sample results."""

    total_em = 0
    total_f1 = 0.0
    total_substring = 0
    total_nll = 0.0
    total_nll_tokens = 0
    per_sample = []

    for i, sample in enumerate(samples):
        prompt_ids = sample['prompt_ids'].to(device)
        seq_len = prompt_ids.shape[1]

        # Reset H2O state for each sample
        reset_h2o_state(model)

        # Generate answer
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # Extract generated text (only the new tokens)
        generated_ids = output_ids[0, seq_len:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Truncate at newline (model may generate extra)
        prediction_clean = prediction.split('\n')[0].strip()

        # Score
        scores = score_prediction(prediction_clean, sample['answers'])
        total_em += scores['exact_match']
        total_f1 += scores['f1']
        total_substring += scores['substring_match']

        # Compute perplexity on the prompt (prefill-then-decode)
        reset_h2o_state(model)
        with torch.no_grad():
            # Prefill first half
            prefill_len = seq_len // 2
            outputs = model(prompt_ids[:, :prefill_len], use_cache=True)
            past_kv = outputs.past_key_values

            chunk_nll = 0.0
            chunk_tokens = 0

            # NLL for first decode token
            logits = outputs.logits[:, -1, :]
            target = prompt_ids[:, prefill_len]
            loss = F.cross_entropy(logits.float(), target)
            chunk_nll += loss.item()
            chunk_tokens += 1

            # Decode rest
            for j in range(prefill_len, seq_len - 1):
                token = prompt_ids[:, j:j+1]
                outputs = model(token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                target = prompt_ids[:, j + 1]
                loss = F.cross_entropy(logits.float(), target)
                chunk_nll += loss.item()
                chunk_tokens += 1

        sample_ppl = torch.exp(torch.tensor(chunk_nll / chunk_tokens)).item()
        total_nll += chunk_nll
        total_nll_tokens += chunk_tokens

        per_sample.append({
            'question': sample['question'],
            'gold_answers': sample['answers'],
            'prediction': prediction_clean,
            'exact_match': scores['exact_match'],
            'f1': scores['f1'],
            'substring_match': scores['substring_match'],
            'perplexity': round(sample_ppl, 2),
            'gold_position': sample['gold_passage_position'],
            'num_tokens': sample['num_context_tokens'],
            'num_passages': sample['num_passages'],
        })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            em_so_far = total_em / (i + 1) * 100
            f1_so_far = total_f1 / (i + 1) * 100
            sub_so_far = total_substring / (i + 1) * 100
            print(f"  [{i+1}/{len(samples)}] EM={em_so_far:.1f}% F1={f1_so_far:.1f}% "
                  f"Sub={sub_so_far:.1f}%", flush=True)

    n = len(samples)
    avg_ppl = torch.exp(torch.tensor(total_nll / total_nll_tokens)).item() if total_nll_tokens > 0 else 0

    metrics = {
        'exact_match': round(total_em / n * 100, 1),
        'f1': round(total_f1 / n * 100, 1),
        'substring_match': round(total_substring / n * 100, 1),
        'perplexity': round(avg_ppl, 2),
        'num_samples': n,
    }

    # Breakdown by gold passage position
    for pos in ['beginning', 'middle', 'end']:
        pos_samples = [s for s in per_sample if s['gold_position'] == pos]
        if pos_samples:
            metrics[f'em_{pos}'] = round(sum(s['exact_match'] for s in pos_samples) / len(pos_samples) * 100, 1)
            metrics[f'f1_{pos}'] = round(sum(s['f1'] for s in pos_samples) / len(pos_samples) * 100, 1)
            metrics[f'sub_{pos}'] = round(sum(s['substring_match'] for s in pos_samples) / len(pos_samples) * 100, 1)
            metrics[f'n_{pos}'] = len(pos_samples)

    return metrics, per_sample


def evaluate_config(model_name, tokenizer, samples, heavy_ratio=None,
                    recent_ratio=None, max_new_tokens=50):
    """Evaluate a single config (baseline or H2O). Returns results dict."""
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
    metrics, per_sample = evaluate_qa(model, tokenizer, samples, device,
                                       max_new_tokens=max_new_tokens)
    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()

    total_tokens = sum(s['num_context_tokens'] for s in samples)
    throughput = total_tokens / elapsed

    del model
    torch.cuda.empty_cache()

    result = {
        'label': label,
        'heavy_ratio': heavy_ratio,
        'recent_ratio': recent_ratio,
        **metrics,
        'peak_memory_mb': round(peak_mem, 1),
        'time_seconds': round(elapsed, 1),
        'throughput_tok_per_sec': round(throughput, 1),
    }

    print(f"\n  Exact Match: {metrics['exact_match']}%")
    print(f"  F1:          {metrics['f1']}%")
    print(f"  Substring:   {metrics['substring_match']}%")
    print(f"  Perplexity:  {metrics['perplexity']}")
    print(f"  Peak Memory: {peak_mem:.1f} MB")
    print(f"  Time:        {elapsed:.1f}s", flush=True)

    return result, per_sample


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Long-Context QA Benchmark for H2O KV Cache Evaluation")
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--target_context_tokens", type=int, default=1800,
                        help="Target number of tokens per QA context")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--h2o_configs", type=str, default="0.1,0.2,0.3,0.5",
                        help="Comma-separated ratios for heavy and recent")
    parser.add_argument("--output_path", type=str,
                        default="./benchmark_results/longcontext_qa_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ratios = [float(r) for r in args.h2o_configs.split(",")]

    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}, Target tokens: {args.target_context_tokens}")
    print(f"H2O configs: {ratios}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = prepare_longcontext_qa(
        tokenizer,
        num_samples=args.num_samples,
        target_context_tokens=args.target_context_tokens,
        seed=args.seed,
    )

    if not samples:
        print("ERROR: No samples prepared. Check dataset availability.")
        return

    all_results = []
    all_per_sample = {}

    # Baseline
    result, per_sample = evaluate_config(
        args.model_name, tokenizer, samples,
        max_new_tokens=args.max_new_tokens)
    all_results.append(result)
    all_per_sample['baseline'] = per_sample

    # H2O configs
    for ratio in ratios:
        result, per_sample = evaluate_config(
            args.model_name, tokenizer, samples,
            heavy_ratio=ratio, recent_ratio=ratio,
            max_new_tokens=args.max_new_tokens)
        all_results.append(result)
        all_per_sample[f'h2o_{ratio}'] = per_sample

    # Print results table
    baseline = all_results[0]
    print(f"\n{'='*90}")
    print(f"RESULTS: {args.model_name}")
    print(f"Long-Context QA (SQuAD), {len(samples)} samples, "
          f"~{args.target_context_tokens} tokens/sample")
    print(f"{'='*90}")
    print(f"{'Config':<18} {'EM%':>6} {'F1%':>6} {'Sub%':>6} "
          f"{'PPL':>8} {'Mem(MB)':>10} {'Time(s)':>10}")
    print(f"{'-'*72}")

    for r in all_results:
        print(f"{r['label']:<18} {r['exact_match']:>6.1f} {r['f1']:>6.1f} "
              f"{r['substring_match']:>6.1f} {r['perplexity']:>8.2f} "
              f"{r['peak_memory_mb']:>10.1f} {r['time_seconds']:>10.1f}")

    print(f"{'='*90}")

    # Position breakdown
    print(f"\n{'Position Breakdown (Substring Match %):':}")
    print(f"{'Config':<18} {'Begin':>8} {'Middle':>8} {'End':>8}")
    print(f"{'-'*44}")
    for r in all_results:
        beg = r.get('sub_beginning', '-')
        mid = r.get('sub_middle', '-')
        end = r.get('sub_end', '-')
        beg_s = f"{beg:.1f}" if isinstance(beg, (int, float)) else beg
        mid_s = f"{mid:.1f}" if isinstance(mid, (int, float)) else mid
        end_s = f"{end:.1f}" if isinstance(end, (int, float)) else end
        print(f"{r['label']:<18} {beg_s:>8} {mid_s:>8} {end_s:>8}")
    print()

    # Save JSON
    output = {
        'model': args.model_name,
        'dataset': 'SQuAD (long-context)',
        'num_samples': len(samples),
        'target_context_tokens': args.target_context_tokens,
        'max_new_tokens': args.max_new_tokens,
        'summary': all_results,
        'per_sample': all_per_sample,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output_path}")

    # Print a few example predictions
    print(f"\n{'='*60}")
    print("Example predictions (Baseline vs most aggressive H2O):")
    print(f"{'='*60}")
    baseline_samples = all_per_sample.get('baseline', [])
    h2o_key = f'h2o_{ratios[0]}' if ratios else None
    h2o_samples = all_per_sample.get(h2o_key, []) if h2o_key else []

    for i in range(min(5, len(baseline_samples))):
        bs = baseline_samples[i]
        print(f"\nQ: {bs['question']}")
        print(f"Gold: {bs['gold_answers'][0]}")
        print(f"Baseline: {bs['prediction']} (EM={bs['exact_match']}, F1={bs['f1']:.2f})")
        if i < len(h2o_samples):
            hs = h2o_samples[i]
            print(f"H2O({ratios[0]}): {hs['prediction']} (EM={hs['exact_match']}, F1={hs['f1']:.2f})")


if __name__ == "__main__":
    main()
