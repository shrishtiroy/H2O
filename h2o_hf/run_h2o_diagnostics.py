#!/usr/bin/env python
"""
Compare H2O performance on standard LLM tasks: text generation, summarization, streaming.
"""

import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def load_model(model_name, enable_h2o=False, heavy_ratio=0.1, recent_ratio=0.1):
    """Load a model with optional H2O."""
    print(f"\nLoading {model_name} ({'H2O' if enable_h2o else 'baseline'})...")
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    if enable_h2o:
        config.heavy_ratio = heavy_ratio
        config.recent_ratio = recent_ratio
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if enable_h2o:
        # Auto-detect model type and apply H2O
        model_type = model.config.model_type.lower()
        
        if "llama" in model_type:
            from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent
            print(f"Applying H2O for LLaMA...")
            model = convert_kvcache_llama_heavy_recent(model, config)
        elif "opt" in model_type:
            from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent
            print(f"Applying H2O for OPT...")
            model = convert_kvcache_opt_heavy_recent(model, config)
        elif "gpt_neox" in model_type or "gptneox" in model_type:
            from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent
            print(f"Applying H2O for GPTNeoX...")
            model = convert_kvcache_gpt_neox_heavy_recent(model, config)
        else:
            print(f"H2O not supported for {model_type}")
    
    model.eval()
    return model

def test_text_generation(model, tokenizer, prompt="The future of AI is", max_tokens=50):
    """Test text generation."""
    print(f"\n{'='*70}")
    print(f"Text Generation Test")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*70}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_tokens = outputs.shape[1]
    tokens_per_sec = (output_tokens - inputs['input_ids'].shape[1]) / elapsed
    
    print(f"Response: {response[:300]}...")
    print(f"Time: {elapsed:.2f}s, Tokens/sec: {tokens_per_sec:.1f}")
    
    return {
        "time": elapsed,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "response_preview": response[:100]
    }

def test_long_context(model, tokenizer, context_len=2000, query_len=100):
    """Test model behavior with long context."""
    print(f"\n{'='*70}")
    print(f"Long Context Test")
    print(f"Context length: {context_len}, Query length: {query_len}")
    print(f"{'='*70}")
    
    # Create a long context
    prompt_tokens = ["word"] * context_len
    prompt = " ".join(prompt_tokens)
    prompt += "\nQuestion: What is the main topic?"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]
    
    print(f"Input length: {input_len} tokens")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.time() - start_time
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    print(f"Response: {response[:200]}...")
    print(f"Generation time: {elapsed:.2f}s")
    
    return {
        "time": elapsed,
        "input_length": input_len,
        "response_preview": response[:100]
    }

def main():
    # Test with a smaller, faster model that has H2O support
    # Supported H2O models: LLaMA, OPT, GPTNeoX/Pythia, Qwen
    # Using TinyLlama-1.1B for speed - small LLaMA-based model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Alternative smaller models:
    # - "facebook/opt-125m" (OPT, has H2O but may have signature issues)
    # - "EleutherAI/pythia-160m" (Pythia, GPTNeoX-based, has import issues)
    # - "facebook/opt-350m" (slightly larger OPT)
    # - "meta-llama/Llama-2-7b-hf" (requires access, if available use this)
    
    # Load baseline and H2O versions
    model_baseline = load_model(model_name, enable_h2o=False)
    
    # Note: H2O support depends on the model architecture
    # For GPT2, we'll skip H2O since it's not in our supported list
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run tests
    results = {}
    
    print("\n" + "="*70)
    print("BASELINE (No H2O)")
    print("="*70)
    
    results['baseline'] = {
        'text_generation': test_text_generation(model_baseline, tokenizer),
        'long_context': test_long_context(model_baseline, tokenizer, context_len=500),
    }
    
    # Try to test H2O if supported
    print("\n" + "="*70)
    print("Testing H2O Support")
    print("="*70)
    
    model_type = model_baseline.config.model_type.lower()
    # Check if model type matches any supported H2O architecture
    is_supported = any(t in model_type for t in ["llama", "opt", "gpt_neox", "gptneox"])
    
    if is_supported:
        try:
            model_h2o = load_model(model_name, enable_h2o=True, heavy_ratio=0.1, recent_ratio=0.1)
            results['h2o'] = {
                'text_generation': test_text_generation(model_h2o, tokenizer),
                'long_context': test_long_context(model_h2o, tokenizer, context_len=500),
            }
        except Exception as e:
            print(f"Failed to load H2O: {e}")
            results['h2o'] = {'error': str(e)}
    else:
        print(f"H2O not supported for {model_type}")
        print(f"Supported model types: llama, opt, gpt_neox")
        results['h2o'] = {'error': f'Model type {model_type} not supported'}
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(json.dumps(results, indent=2, default=str))
    
    # Save results
    with open("h2o_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to h2o_diagnostic_results.json")

if __name__ == "__main__":
    main()
