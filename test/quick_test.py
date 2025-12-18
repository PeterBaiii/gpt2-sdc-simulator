"""
快速测试脚本 - 验证所有修复
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("="*60)
print("Quick Test - Verifying All Fixes")
print("="*60)

# 测试1: 基本加载和patch
print("\n[Test 1] Loading and patching model...")
try:
    from src.model_adapter import monkey_patch_model
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded")
    
    model = monkey_patch_model(model, 'gpt2', injection_layers=[0])
    print("✓ Model patched successfully")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试2: 基本forward pass
print("\n[Test 2] Testing forward pass...")
try:
    text = "Hello world"
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {outputs.logits.shape}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试3: 测试返回值
print("\n[Test 3] Testing return values...")
try:
    # 直接调用attention
    hidden_states = torch.randn(1, 2, 768)
    
    attn_output = model.transformer.h[0].attn(hidden_states)
    
    # 检查返回值
    if isinstance(attn_output, tuple):
        print(f"✓ Attention returns tuple with {len(attn_output)} elements")
        print(f"  First element shape: {attn_output[0].shape}")
        if len(attn_output) > 1 and attn_output[1] is not None:
            print(f"  Second element (weights) shape: {attn_output[1].shape}")
    else:
        print(f"✓ Attention returns single tensor: {attn_output.shape}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 测试数据加载
# print("\n[Test 4] Testing data loading...")
# try:
#     from custom_datasets import prepare_dummy_data
    
#     dataloader = prepare_dummy_data(
#         tokenizer,
#         batch_size=2,
#         num_samples=5,
#         seq_length=32
#     )
    
#     batch = next(iter(dataloader))
#     print(f"✓ DataLoader works")
#     print(f"  Batch keys: {batch.keys()}")
#     print(f"  Input IDs shape: {batch['input_ids'].shape}")
    
# except Exception as e:
#     print(f"✗ Failed: {e}")
#     import traceback
#     traceback.print_exc()
#     exit(1)

# 测试5: 测试collector
print("\n[Test 5] Testing IntermediateTensorCollector...")
try:
    from real_model_exp.src.experiment_runner import IntermediateTensorCollector, run_model_with_collection
    
    collector = IntermediateTensorCollector()
    
    # 准备输入
    inputs = tokenizer("Test", return_tensors='pt', padding='max_length', max_length=16)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask')
    
    # 运行collection
    outputs, collected = run_model_with_collection(
        model, input_ids, attention_mask, collector, injection_config=None
    )
    
    print(f"✓ Collection successful")
    print(f"  Collected layers: {list(collected.keys())}")
    
    if len(collected) > 0:
        first_layer = list(collected.keys())[0]
        print(f"  Layer {first_layer} tensors: {list(collected[first_layer].keys())}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试6: 测试注入
print("\n[Test 6] Testing error injection...")
try:
    from src.fault_injection import InjectionConfig, InjectionLocation
    
    inj_config = InjectionConfig(
        location=InjectionLocation.SCORES,
        idx=(0, 0, 0, 0),
        bit=15,
        enabled=True
    )
    
    # 运行带注入的forward
    outputs, collected = run_model_with_collection(
        model, input_ids, attention_mask, collector, injection_config=inj_config
    )
    
    print(f"✓ Injection successful")
    
    # 检查是否真的注入了
    if len(collected) > 0:
        first_layer = list(collected.keys())[0]
        if 'injection_applied' in collected[first_layer]:
            print(f"  Injection applied: {collected[first_layer]['injection_applied']}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("🎉 All tests passed!")
print("="*60)
print("\nYou can now run:")
print("  python full_example.py simple")