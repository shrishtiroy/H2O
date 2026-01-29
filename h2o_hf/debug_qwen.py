#!/usr/bin/env python
"""
Debug script to understand Qwen2-VL attention interface.
"""

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
import inspect

model_name = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Find the attention class being used
print("\n=== Attention Module Info ===")
for name, module in model.named_modules():
    if 'layers.0.self_attn' in name and 'language_model' in name:
        print(f"Module: {name}")
        print(f"Class: {type(module).__name__}")
        print(f"Class MRO: {[c.__name__ for c in type(module).__mro__[:5]]}")
        
        # Get forward signature
        sig = inspect.signature(module.forward)
        print(f"\nForward parameters:")
        for param_name, param in sig.parameters.items():
            print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")
        
        # Check return annotation
        print(f"\nReturn annotation: {sig.return_annotation}")
        
        # Look at the source if possible
        try:
            source_lines = inspect.getsourcelines(module.forward)[0]
            # Find the return statement
            for i, line in enumerate(source_lines):
                if 'return ' in line and 'return_' not in line:
                    print(f"\nReturn statement (line ~{i}):")
                    print(f"  {line.strip()}")
        except:
            pass
        
        break

# Check the decoder layer to see how it calls attention
print("\n=== Decoder Layer Info ===")
for name, module in model.named_modules():
    if 'layers.0' in name and 'language_model' in name and not 'self_attn' in name and not 'mlp' in name:
        if hasattr(module, 'forward'):
            print(f"Module: {name}")
            print(f"Class: {type(module).__name__}")
            try:
                source_lines = inspect.getsourcelines(module.forward)[0]
                # Find how self_attn is called
                for i, line in enumerate(source_lines):
                    if 'self_attn' in line:
                        # Print context around this line
                        start = max(0, i-1)
                        end = min(len(source_lines), i+3)
                        print(f"\nself_attn call (lines {start}-{end}):")
                        for j in range(start, end):
                            print(f"  {source_lines[j].rstrip()}")
            except Exception as e:
                print(f"Could not get source: {e}")
            break

print("\n=== Cache Type ===")
# Check what cache type is used
from transformers import DynamicCache
print(f"DynamicCache available: {DynamicCache is not None}")