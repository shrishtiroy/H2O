#!/usr/bin/env python
"""
Debug H2O NaN issue
"""

import torch
from transformers import AutoConfig, Qwen2VLForConditionalGeneration

model_name = "Qwen/Qwen2-VL-2B-Instruct"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.heavy_ratio = 0.2
config.recent_ratio = 0.2
config.sink_token_count = 4
config.min_seq_for_eviction = 1024  # H2O disabled

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Apply H2O conversion
from utils_hh.modify_qwen import convert_kvcache_qwen_heavy_recent
model = convert_kvcache_qwen_heavy_recent(model, config)

# Create simple test inputs
device = next(model.parameters()).device

# Test 1: Check if conversion worked
print("\nTest 1: Model structure after conversion")
for name, module in model.named_modules():
    if 'attention' in name.lower() and 'visual' not in name.lower():
        print(f"  {name}: {type(module).__name__}")
        if hasattr(module, 'h2o_scores'):
            print(f"    - Has h2o_scores attribute")
        break

# Test 2: Simple token generation without cache
print("\nTest 2: Generate without cache")
try:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
    print(f"  Output shape: {outputs.logits.shape}")
    print(f"  Has NaN: {torch.isnan(outputs.logits).any().item()}")
    if torch.isnan(outputs.logits).any():
        print(f"  First 10 logits: {outputs.logits[0, -1, :10]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Simple token generation with cache
print("\nTest 3: Generate with cache")
try:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)
    print(f"  Output shape: {outputs.logits.shape}")
    print(f"  Has NaN: {torch.isnan(outputs.logits).any().item()}")
    if torch.isnan(outputs.logits).any():
        print(f"  First 10 logits: {outputs.logits[0, -1, :10]}")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone!")
