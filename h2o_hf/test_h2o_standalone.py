#!/usr/bin/env python
"""
Detailed debug to trace what's happening in attention.
"""

import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration

model_name = "Qwen/Qwen2-VL-2B-Instruct"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.heavy_ratio = 0.2
config.recent_ratio = 0.2
config.sink_token_count = 4
config.min_seq_for_eviction = 1024  # High threshold so H2O doesn't activate

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Get a reference attention output BEFORE conversion
print("\n=== Testing BEFORE H2O conversion ===")

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
try:
    image = Image.open(requests.get(url, stream=True, timeout=10).raw).convert("RGB")
except:
    import numpy as np
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Hi"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
device = next(model.parameters()).device
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs_before = model(**inputs, use_cache=True, return_dict=True)
    print(f"Logits shape: {outputs_before.logits.shape}")
    print(f"Last token logits (first 10): {outputs_before.logits[0, -1, :10]}")
    next_token_before = torch.argmax(outputs_before.logits[0, -1, :]).item()
    print(f"Predicted next token: {next_token_before} = '{processor.decode([next_token_before])}'")
    
    if outputs_before.past_key_values is not None:
        cache = outputs_before.past_key_values
        print(f"Cache type: {type(cache)}")
        if hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
            print(f"Cache key shape (layer 0): {cache.key_cache[0].shape}")
        if hasattr(cache, 'get_seq_length'):
            print(f"Cache get_seq_length(): {cache.get_seq_length()}")

# Now apply H2O and test again
print("\n=== Applying H2O conversion ===")
from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent
model = convert_kvcache_qwen_heavy_recent(model, config)

print("\n=== Testing AFTER H2O conversion ===")

# Need fresh inputs (can't reuse cache)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs_after = model(**inputs, use_cache=True, return_dict=True)
    print(f"Logits shape: {outputs_after.logits.shape}")
    print(f"Last token logits (first 10): {outputs_after.logits[0, -1, :10]}")
    next_token_after = torch.argmax(outputs_after.logits[0, -1, :]).item()
    print(f"Predicted next token: {next_token_after} = '{processor.decode([next_token_after])}'")
    
    if outputs_after.past_key_values is not None:
        cache = outputs_after.past_key_values
        print(f"Cache type: {type(cache)}")
        if hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
            print(f"Cache key shape (layer 0): {cache.key_cache[0].shape}")
            print(f"Number of layers in cache: {len(cache.key_cache)}")
        if hasattr(cache, 'get_seq_length'):
            print(f"Cache get_seq_length(): {cache.get_seq_length()}")

# Compare
print("\n=== Comparison ===")
print(f"Token BEFORE H2O: {next_token_before} = '{processor.decode([next_token_before])}'")
print(f"Token AFTER H2O:  {next_token_after} = '{processor.decode([next_token_after])}'")

if next_token_before == next_token_after:
    print("✓ Same predicted token - attention seems to work!")
else:
    print("✗ Different predicted token - something is wrong")
    
# Check if logits are similar
logits_diff = (outputs_before.logits[0, -1, :] - outputs_after.logits[0, -1, :]).abs().mean()
print(f"Mean absolute logits difference: {logits_diff.item():.6f}")

if logits_diff < 0.1:
    print("✓ Logits are similar")
else:
    print("✗ Logits differ significantly - attention output is different")