#!/usr/bin/env python
"""
VQA Benchmark: Baseline vs H2O comparison on Qwen2-VL.
Uses VQAv2 validation set with proper VQA accuracy scoring.

Usage:
    # Default: 100 samples, Qwen2-VL-2B
    python run_vqa_benchmark.py

    # Custom sample count and model
    python run_vqa_benchmark.py --num_samples 500 --model_name Qwen/Qwen2-VL-7B-Instruct

    # Custom H2O ratios
    python run_vqa_benchmark.py --heavy_ratio 0.2 --recent_ratio 0.2
"""
import argparse
import os
import time
import json

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent


def vqa_accuracy(prediction: str, answers: list) -> float:
    """VQA accuracy: min(#humans that said answer / 3, 1)"""
    pred = prediction.strip().lower()
    for prefix in ['the answer is ', 'answer: ', 'it is ', 'they are ', 'there are ']:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
    pred = pred.strip().rstrip('.')

    matches = 0
    for ans_dict in answers:
        gt = ans_dict['answer'].lower().strip()
        if pred == gt or gt in pred or pred in gt:
            matches += 1

    return min(matches / 3.0, 1.0)


def load_model(model_name, enable_h2o=False, heavy_ratio=0.1, recent_ratio=0.1):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.heavy_ratio = heavy_ratio
    config.recent_ratio = recent_ratio

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
    )

    if enable_h2o:
        checkpoint = model.state_dict()
        model = convert_kvcache_qwen_heavy_recent(model, config)
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint

    model.half().to('cuda').eval()
    return model, processor


def evaluate(model, processor, dataset, num_samples, label=""):
    total_acc = 0.0
    total = 0
    start_time = time.time()

    for i, sample in enumerate(dataset):
        image = sample['image']
        question = sample['question']
        answers = sample['answers']

        if image is None:
            continue

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': f'{question}\nAnswer the question using a single word or phrase.'}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt')
        inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        input_len = inputs['input_ids'].shape[1]
        new_tokens = output_ids[0, input_len:]
        response = processor.decode(new_tokens, skip_special_tokens=True).strip()

        acc = vqa_accuracy(response, answers)
        total_acc += acc
        total += 1

        if total % 20 == 0:
            avg = total_acc / total * 100
            elapsed = time.time() - start_time
            print(f'  [{label}] {total}/{num_samples} samples, '
                  f'accuracy: {avg:.1f}%, '
                  f'time: {elapsed:.0f}s', flush=True)

    elapsed = time.time() - start_time
    avg_acc = total_acc / total * 100 if total > 0 else 0
    return avg_acc, total, elapsed


def main():
    parser = argparse.ArgumentParser(description='VQA Benchmark: Baseline vs H2O')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--heavy_ratio', type=float, default=0.1)
    parser.add_argument('--recent_ratio', type=float, default=0.1)
    parser.add_argument('--output_path', type=str, default='./vqa_results/vqa_benchmark_results.json')
    args = parser.parse_args()

    print(f"Loading VQAv2 validation set ({args.num_samples} samples)...", flush=True)
    dataset = load_dataset('lmms-lab/VQAv2', split=f'validation[:{args.num_samples}]')

    # --- Baseline ---
    print(f"\n{'='*60}")
    print(f"Running BASELINE (full KV cache) - {args.model_name}")
    print(f"{'='*60}", flush=True)
    model, processor = load_model(args.model_name, enable_h2o=False)
    baseline_acc, baseline_total, baseline_time = evaluate(
        model, processor, dataset, args.num_samples, "Baseline")
    del model
    torch.cuda.empty_cache()

    # --- H2O ---
    print(f"\n{'='*60}")
    print(f"Running H2O (heavy={args.heavy_ratio}, recent={args.recent_ratio})")
    print(f"{'='*60}", flush=True)
    model, processor = load_model(
        args.model_name, enable_h2o=True,
        heavy_ratio=args.heavy_ratio, recent_ratio=args.recent_ratio)
    h2o_acc, h2o_total, h2o_time = evaluate(
        model, processor, dataset, args.num_samples, "H2O")
    del model
    torch.cuda.empty_cache()

    # --- Results ---
    h2o_label = f'H2O ({args.heavy_ratio}/{args.recent_ratio})'
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model_name} on VQAv2 ({args.num_samples} samples)")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Time':>10} {'Samples':>10}")
    print(f"{'-'*55}")
    print(f"{'Baseline':<25} {baseline_acc:>9.1f}% {baseline_time:>9.0f}s {baseline_total:>10}")
    print(f"{h2o_label:<25} {h2o_acc:>9.1f}% {h2o_time:>9.0f}s {h2o_total:>10}")
    print(f"{'-'*55}")
    print(f"{'Difference':<25} {h2o_acc - baseline_acc:>+9.1f}%")
    print(f"{'='*60}")

    # --- Save ---
    results = {
        'model': args.model_name,
        'dataset': 'VQAv2',
        'num_samples': args.num_samples,
        'baseline': {'accuracy': baseline_acc, 'time': baseline_time, 'total': baseline_total},
        'h2o': {'accuracy': h2o_acc, 'time': h2o_time, 'total': h2o_total,
                'heavy_ratio': args.heavy_ratio, 'recent_ratio': args.recent_ratio},
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == '__main__':
    main()
