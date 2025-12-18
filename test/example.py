"""
使用示例 - 展示如何使用模块化框架进行实验
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader

# 导入我们的模块
from src.fault_injection import InjectionConfig, InjectionLocation
from src.bounds_computation import compute_attention_bounds, detect_violation
from src.experiment_config import ExperimentConfig, ParameterSweepConfig, ConfigTemplates
# from experiment_runner import ExperimentRunner, run_parameter_sweep
from src.model_adapter import monkey_patch_model


# ============ 示例1: 单次实验 ============
def example_single_run():
    """单次实验示例"""
    
    print("=" * 60)
    print("示例1: 单次实验")
    print("=" * 60)
    
    # 1. 创建配置
    config = ExperimentConfig(
        exp_name="single_run_demo",
        model_name="gpt2",
        batch_size=4,
        num_samples=10,
        injection_enabled=True,
        injection_location="scores",
        injection_idx=(0, 0, 2, 7),
        injection_bit=15,
        seed=42
    )
    
    # 保存配置
    config.save()
    print(f"Config saved to: {config.save_dir}/{config.exp_id}/config.json")
    
    # 2. 加载模型
    print("\nLoading model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = monkey_patch_model(model, 'gpt2')
    model.eval()
    
    # 3. 准备数据
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    
    # 4. Baseline运行(不注入错误)
    print("\nRunning baseline...")
    with torch.no_grad():
        baseline_outputs = model(**inputs, output_hidden_states=True)
        baseline_loss = baseline_outputs.loss if hasattr(baseline_outputs, 'loss') else None
    
    print(f"Baseline loss: {baseline_loss}")
    
    # 5. 注入错误运行
    print("\nRunning with injection...")
    injection_cfg = InjectionConfig(
        location=InjectionLocation.SCORES,
        idx=config.injection_idx,
        bit=config.injection_bit,
        enabled=True
    )
    
    # TODO: 实际注入需要在模型内部实现
    # 这里仅作演示
    
    print("\n" + "=" * 60)
    print("示例1完成")
    print("=" * 60 + "\n")


# ============ 示例2: 参数扫描 ============
def example_parameter_sweep():
    """参数扫描示例"""
    
    print("=" * 60)
    print("示例2: 参数扫描")
    print("=" * 60)
    
    # 使用预定义模板
    sweep_config = ConfigTemplates.bit_sweep()
    
    print(f"Total configurations: {sweep_config.get_num_configs()}")
    
    # 生成所有配置
    configs = sweep_config.generate_configs()
    
    print("\n前3个配置:")
    for i, cfg in enumerate(configs[:3]):
        print(f"{i+1}. {cfg.exp_id}")
        print(f"   Seed: {cfg.seed}, Bit: {cfg.injection_bit}")
    
    print("\n" + "=" * 60)
    print("示例2完成")
    print("=" * 60 + "\n")


# ============ 示例3: 自定义参数扫描 ============
def example_custom_sweep():
    """自定义参数扫描示例"""
    
    print("=" * 60)
    print("示例3: 自定义参数扫描")
    print("=" * 60)
    
    # 创建基础配置
    base_config = ExperimentConfig(
        exp_name="custom_sweep",
        model_name="gpt2",
        batch_size=4,
        num_samples=50
    )
    
    # 定义要扫描的参数
    sweep_config = ParameterSweepConfig(
        base_config=base_config,
        sweep_params={
            'seed': [42, 123, 456],
            'injection_bit': [0, 7, 15, 23, 31],
            'injection_location': ['scores', 'weights', 'out'],
        }
    )
    
    print(f"Total configurations: {sweep_config.get_num_configs()}")
    print(f"= 3 seeds × 5 bits × 3 locations = {3*5*3} configs")
    
    configs = sweep_config.generate_configs()
    
    print("\n示例配置:")
    for cfg in configs[::15]:  # 每15个显示一个
        print(f"  - {cfg.exp_id}")
    
    print("\n" + "=" * 60)
    print("示例3完成")
    print("=" * 60 + "\n")


# ============ 示例4: 边界计算 ============
def example_bounds_computation():
    """边界计算示例"""
    
    print("=" * 60)
    print("示例4: 边界计算与检测")
    print("=" * 60)
    
    # 创建模拟的注意力分数和权重
    B, H, T = 2, 4, 8
    d = 16
    
    scores = torch.randn(B, H, T, T) * 2
    p = torch.softmax(scores, dim=-1)
    
    print(f"Scores shape: {scores.shape}")
    print(f"Weights shape: {p.shape}")
    
    # 计算边界
    from src.bounds_computation import compute_attention_bounds
    
    bounds = compute_attention_bounds(scores, p, d)
    
    print("\n边界统计:")
    print(f"Epsilon - mean: {bounds.epsilon.mean():.6f}, std: {bounds.epsilon.std():.6f}")
    print(f"Lower bound - mean: {bounds.lower1.mean():.6f}")
    print(f"Upper bound - mean: {bounds.upper.mean():.6f}")
    
    # 检查不等式
    check_result = bounds.check_inequalities()
    print("\n不等式检查:")
    for key, value in check_result.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # 模拟注入后的epsilon
    injected_epsilon = bounds.epsilon + torch.randn_like(bounds.epsilon) * 0.1
    
    # 检测违反
    from src.bounds_computation import detect_violation
    
    violation_result = detect_violation(bounds, injected_epsilon)
    
    print("\n违反检测:")
    
    assert isinstance(violation_result['injection_violations'], dict), "detection['injection_violations'] should be a dict"
    
    if violation_result['injection_violations']['any_violated']:
        print(f"  检测到违反!")
        print(f"  Lower violations: {violation_result['injection_violations']['num_lower_violations']}")
        print(f"  Upper violations: {violation_result['injection_violations']['num_upper_violations']}")
    else:
        print(f"  未检测到违反")
    
    print("\n" + "=" * 60)
    print("示例4完成")
    print("=" * 60 + "\n")


# ============ 示例5: 完整工作流 ============
def example_full_workflow():
    """完整工作流示例(伪代码)"""
    
    print("=" * 60)
    print("示例5: 完整工作流(概念演示)")
    print("=" * 60)
    
    print("""
完整工作流步骤:

1. 配置实验
   - 创建ExperimentConfig
   - 设置模型、数据、注入参数
   
2. 加载模型和数据
   - 使用HuggingFace加载预训练模型
   - 使用monkey_patch_model进行hack
   - 准备数据集和DataLoader
   
3. 运行实验
   - 创建ExperimentRunner
   - 运行baseline(无注入)
   - 运行injected(有注入)
   - 捕获中间张量(q, k, v, scores, weights)
   
4. 计算边界
   - 使用compute_attention_bounds计算理论边界
   - 比较baseline和injected的epsilon值
   
5. 检测违反
   - 使用detect_violation检查是否超出边界
   - 记录违反位置和统计信息
   
6. 保存结果
   - 自动保存配置、结果、日志
   - 生成汇总统计
   
7. 参数扫描(可选)
   - 使用ParameterSweepConfig定义扫描范围
   - 批量运行多个配置
   - 汇总所有结果
    """)
    
    print("=" * 60)
    print("示例5完成")
    print("=" * 60 + "\n")


# ============ 主函数 ============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPU静默数据错误检测框架 - 使用示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_single_run()
    example_parameter_sweep()
    example_custom_sweep()
    example_bounds_computation()
    example_full_workflow()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60 + "\n")
    
    print("下一步:")
    print("1. 完善model_adapter.py中的模型hack实现")
    print("2. 在experiment_runner.py中实现完整的实验逻辑")
    print("3. 准备真实数据集(WikiText, OpenWebText等)")
    print("4. 运行大规模参数扫描实验")
    print("5. 分析结果,优化边界公式\n")