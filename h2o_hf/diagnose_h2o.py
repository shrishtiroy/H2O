#!/usr/bin/env python
"""
Diagnose H2O behavior with increasing sequence lengths.
"""

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import requests

model_name = "Qwen/Qwen2-VL-2B-Instruct"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.heavy_ratio = 0.1
config.recent_ratio = 0.1
config.sink_token_count = 4
config.min_seq_for_eviction = 100  # Lower threshold to trigger H2O faster

print("Loading model WITHOUT H2O...")
model_no_h2o = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model_no_h2o.eval()

print("Loading model WITH H2O...")
model_h2o = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent
model_h2o = convert_kvcache_qwen_heavy_recent(model_h2o, config)
model_h2o.eval()

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Get test image
try:
    url = "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png"
    image = Image.open(requests.get(url, stream=True, timeout=10).raw).convert("RGB")
except:
    import numpy as np
    image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

device = next(model_h2o.parameters()).device

print("\n" + "="*70)
print("TEST: Compare outputs at different sequence lengths")
print("="*70)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print(f"\nInitial KV cache sequence length: {inputs['input_ids'].shape[1]}")

# Generate with both models
print("\n" + "-"*70)
print("Generating WITHOUT H2O...")
print("-"*70)

with torch.no_grad():
    output_no_h2o = model_no_h2o.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

response_no_h2o = processor.decode(output_no_h2o.sequences[0], skip_special_tokens=True)
print(f"Output length: {output_no_h2o.sequences.shape[1]} tokens")
print(f"Response preview: {response_no_h2o[-200:]}")

# Reset inputs for H2O model
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("\n" + "-"*70)
print("Generating WITH H2O...")
print("-"*70)

with torch.no_grad():
    output_h2o = model_h2o.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

response_h2o = processor.decode(output_h2o.sequences[0], skip_special_tokens=True)
print(f"Output length: {output_h2o.sequences.shape[1]} tokens")
print(f"Response preview: {response_h2o[-200:]}")

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

seq1 = output_no_h2o.sequences[0]
seq2 = output_h2o.sequences[0]

print(f"\nOutput lengths: WITHOUT H2O={len(seq1)}, WITH H2O={len(seq2)}")
print(f"Unique tokens WITHOUT H2O: {len(torch.unique(seq1))}")
print(f"Unique tokens WITH H2O: {len(torch.unique(seq2))}")

# Compare up to minimum length
min_len = min(len(seq1), len(seq2))
num_matching = (seq1[:min_len] == seq2[:min_len]).sum().item()
print(f"Matching tokens (first {min_len}): {num_matching}/{min_len} ({100*num_matching/min_len:.1f}%)")

if num_matching < min_len:
    # Find where they first diverge
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            print(f"\nFirst divergence at position {i}:")
            print(f"  WITHOUT H2O: {seq1[max(0, i-5):i+5].tolist()}")
            print(f"  WITH H2O:    {seq2[max(0, i-5):i+5].tolist()}")
            break

print("\n" + "="*70)
print("FULL RESPONSES")
print("="*70)

print("\nWITHOUT H2O:")
print(response_no_h2o)

print("\nWITH H2O:")
print(response_h2o)
