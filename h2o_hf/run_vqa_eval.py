#!/usr/bin/env python
# coding=utf-8
"""
VQA Evaluation script for Qwen2-VL / Qwen3-VL with H2O (Heavy-Hitter Oracle).

This script supports:
1. lmms-eval benchmark evaluation (VQAv2, TextVQA, GQA, etc.)
2. Manual single-image testing for sanity checks

Usage:
    # Run VQA benchmark with H2O
    python run_vqa_eval.py \
        --model_name Qwen/Qwen2-VL-7B-Instruct \
        --tasks vqav2_val_lite \
        --enable_h2o \
        --heavy_ratio 0.1 \
        --recent_ratio 0.1

    # Quick sanity check with a single image
    python run_vqa_eval.py \
        --model_name Qwen/Qwen2-VL-7B-Instruct \
        --sanity_check \
        --enable_h2o
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
import requests

from transformers import AutoConfig, AutoProcessor


# Try importing Qwen model classes
try:
    from transformers import Qwen2VLForConditionalGeneration
    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False
    Qwen3VLForConditionalGeneration = None

from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent


def get_model_class(model_name: str):
    """Determine the appropriate model class based on model name."""
    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        if not HAS_QWEN3VL:
            raise ImportError("Qwen3VLForConditionalGeneration not available. Update transformers.")
        return Qwen3VLForConditionalGeneration
    elif "qwen2" in model_name_lower or "qwen" in model_name_lower:
        if not HAS_QWEN2VL:
            raise ImportError("Qwen2VLForConditionalGeneration not available. Update transformers.")
        return Qwen2VLForConditionalGeneration
    else:
        # Default to Qwen2VL
        if HAS_QWEN2VL:
            return Qwen2VLForConditionalGeneration
        elif HAS_QWEN3VL:
            return Qwen3VLForConditionalGeneration
        else:
            raise ImportError("No Qwen VL model class available.")


def load_model(args):
    """Load the Qwen VL model with optional H2O."""
    print(f"Loading model: {args.model_name}")

    # Load config
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio

    # Determine model class
    model_class = get_model_class(args.model_name)
    print(f"Using model class: {model_class.__name__}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.enable_h2o:
        # For H2O: load on CPU first, convert, then move to device
        print(f"Enabling H2O with heavy_ratio={args.heavy_ratio}, recent_ratio={args.recent_ratio}")

        model_kwargs = {
            "torch_dtype": torch.float16 if args.fp16 else torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if args.cache_dir:
            model_kwargs["cache_dir"] = args.cache_dir

        # Check for quantization
        if args.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
                print("Using 4-bit quantization")
            except ImportError:
                print("WARNING: bitsandbytes not installed, skipping 4-bit quantization")
        elif args.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["device_map"] = "auto"
                print("Using 8-bit quantization")
            except ImportError:
                print("WARNING: bitsandbytes not installed, skipping 8-bit quantization")

        # Load model
        model = model_class.from_pretrained(args.model_name, **model_kwargs)

        # For quantized models, we can't easily do the checkpoint dance
        # Instead, convert in-place and the weights should be preserved
        if args.load_in_4bit or args.load_in_8bit:
            # Quantized models: convert without state_dict reload
            # The Linear layers are already quantized, we just replace the attention logic
            print("Converting quantized model (weights preserved in-place)...")
            model = convert_kvcache_qwen_heavy_recent(model, config)
        else:
            # Non-quantized: use state_dict approach but more memory efficient
            print("Saving state dict...")
            checkpoint = model.state_dict()  # Don't deepcopy, just reference

            print("Converting model...")
            model = convert_kvcache_qwen_heavy_recent(model, config)

            print("Restoring weights...")
            model.load_state_dict(checkpoint, strict=False)
            del checkpoint  # Free memory

            # Move to device AFTER conversion
            print(f"Moving to {device}...")
            if args.fp16:
                model = model.half()
            model = model.to(device)

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("H2O enabled successfully")

    else:
        # Without H2O: can use device_map for automatic placement
        model_kwargs = {
            "torch_dtype": torch.float16 if args.fp16 else torch.float32,
            "trust_remote_code": True,
        }
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        if args.cache_dir:
            model_kwargs["cache_dir"] = args.cache_dir

        model = model_class.from_pretrained(args.model_name, **model_kwargs)

        # Move to device if not using device_map
        if not args.device_map:
            model = model.to(device)
    # Add this right after model loading in run_vqa_eval.py (after line ~160)
    print("\n=== Checking attention layers ===")
    for name, module in model.named_modules():
        if "attn" in name.lower():
            print(f"{name}: {type(module).__name__}")
            break  # Just show first one
    model.eval()

    return model, processor, config


def run_sanity_check(model, processor, args):
    """Run a quick sanity check with a sample image."""
    print("\n" + "="*60)
    print("Running sanity check...")
    print("="*60)

    # Use provided image or default test image
    if args.image_path:
        print(f"Loading image from: {args.image_path}")
        image = Image.open(args.image_path)
    else:
        # Try multiple fallback URLs
        test_urls = [
            "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
            "https://picsum.photos/400/300",
        ]

        image = None
        for url in test_urls:
            try:
                print(f"Trying test image from: {url}")
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(response.raw).convert("RGB")
                print("Image loaded successfully!")
                break
            except Exception as e:
                print(f"Failed to load from {url}: {e}")
                continue

        if image is None:
            # Create a simple synthetic image as last resort
            print("Creating synthetic test image...")
            import numpy as np
            img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)

    # Default question
    question = args.question if args.question else "Hello. Describe this image in detail."
    print(f"Question: {question}")

    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate
    print("\nGenerating response...")
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        print(f"Output IDs shape: {output_ids.shape}")
        print(f"Output IDs: {output_ids[0, -20:]}")  # Last 20 token IDs
        print(f"Unique tokens in output: {torch.unique(output_ids)}")
    elapsed_time = time.time() - start_time

    # Decode output
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n" + "-"*60)
    print("Response:")
    print("-"*60)
    print(response)
    print("-"*60)
    print(f"Generation time: {elapsed_time:.2f}s")
    print(f"Output tokens: {output_ids.shape[1]}")
    print("="*60 + "\n")

    return response


def run_lmms_eval(model, processor, args):
    """Run lmms-eval benchmark evaluation."""
    try:
        from lmms_eval import evaluator
        from lmms_eval.api.registry import get_model
    except ImportError:
        print("ERROR: lmms-eval not installed. Install with: pip install lmms-eval")
        print("Or: git clone https://github.com/EvolvingLMMs-Lab/lmms-eval && cd lmms-eval && pip install -e .")
        return None

    print("\n" + "="*60)
    print("Running lmms-eval benchmarks")
    print(f"Tasks: {args.tasks}")
    print("="*60)

    # Parse tasks
    task_list = [t.strip() for t in args.tasks.split(",")]

    # lmms-eval expects the model in a specific format
    # We need to create a custom wrapper or use their built-in Qwen2VL support

    # Option 1: Use lmms-eval's built-in model loading with our modified model
    # This requires saving the model first or using a custom model class

    # For now, let's use the direct evaluation approach
    try:
        # Try to use lmms-eval's Qwen2VL wrapper
        from lmms_eval.models.qwen2_vl import Qwen2VL

        # Create a wrapper that uses our H2O model
        class H2OQwen2VL(Qwen2VL):
            def __init__(self, model, processor, batch_size=1):
                self._model = model
                self._processor = processor
                self.batch_size = batch_size
                self._device = next(model.parameters()).device

            @property
            def model(self):
                return self._model

            @property
            def processor(self):
                return self._processor

            @property
            def device(self):
                return self._device

        lm = H2OQwen2VL(model, processor, batch_size=args.batch_size)

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_list,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

    except Exception as e:
        print(f"Failed to use lmms-eval wrapper: {e}")
        print("\nFalling back to manual evaluation...")
        results = run_manual_vqa_eval(model, processor, args, task_list)

    # Save results
    if results and args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, f"results_{args.tasks.replace(',', '_')}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def run_manual_vqa_eval(model, processor, args, task_list):
    """
    Manual VQA evaluation when lmms-eval integration doesn't work.
    This is a simplified evaluation for testing purposes.
    """
    print("\nRunning manual VQA evaluation (simplified)...")

    results = {}

    for task in task_list:
        print(f"\nEvaluating task: {task}")

        # Try to load the task dataset
        try:
            from datasets import load_dataset

            if "vqav2" in task.lower():
                dataset = load_dataset("lmms-lab/VQAv2", split="validation[:100]" if args.limit else "validation")
            elif "textvqa" in task.lower():
                dataset = load_dataset("lmms-lab/textvqa", split="validation[:100]" if args.limit else "validation")
            elif "gqa" in task.lower():
                dataset = load_dataset("lmms-lab/GQA", split="testdev_balanced[:100]" if args.limit else "testdev_balanced")
            else:
                print(f"Unknown task: {task}, skipping...")
                continue

            correct = 0
            total = 0

            for sample in dataset:
                # Extract image and question
                image = sample.get("image")
                question = sample.get("question")
                answer = sample.get("answer", sample.get("answers", [""])[0] if isinstance(sample.get("answers"), list) else "")

                if image is None or question is None:
                    continue

                # Prepare input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt")

                device = next(model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

                response = processor.decode(output_ids[0], skip_special_tokens=True)

                # Simple accuracy check (exact match or contains)
                if isinstance(answer, str) and answer.lower() in response.lower():
                    correct += 1

                total += 1

                if total % 10 == 0:
                    print(f"  Progress: {total} samples, accuracy: {correct/total:.2%}")

            results[task] = {
                "accuracy": correct / total if total > 0 else 0,
                "correct": correct,
                "total": total
            }

        except Exception as e:
            print(f"Failed to evaluate {task}: {e}")
            results[task] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="VQA Evaluation with H2O for Qwen VL models")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for model weights")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model parallelism")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 precision")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")

    # H2O arguments
    parser.add_argument("--enable_h2o", action="store_true",
                        help="Enable H2O (Heavy-Hitter Oracle) for KV cache")
    parser.add_argument("--heavy_ratio", type=float, default=0.1,
                        help="Ratio of heavy hitter tokens to keep")
    parser.add_argument("--recent_ratio", type=float, default=0.1,
                        help="Ratio of recent tokens to keep")

    # Quantization arguments (helps with memory)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit quantization (requires bitsandbytes)")

    # Evaluation arguments
    parser.add_argument("--tasks", type=str, default="vqav2_val_lite",
                        help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per task")
    parser.add_argument("--output_path", type=str, default="./vqa_results",
                        help="Output directory for results")

    # Sanity check arguments
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run a quick sanity check instead of full eval")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to image for sanity check")
    parser.add_argument("--question", type=str, default=None,
                        help="Question for sanity check")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens for generation")

    args = parser.parse_args()

    # Load model
    model, processor, config = load_model(args)

    if args.sanity_check:
        # Run quick sanity check
        run_sanity_check(model, processor, args)
    else:
        # Run full evaluation
        results = run_lmms_eval(model, processor, args)

        if results:
            print("\n" + "="*60)
            print("Evaluation Results:")
            print("="*60)
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
