"""
完整使用示例 - 从加载模型到运行实验的完整流程
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, Dataset as HFDataset

from src.experiment_config import ExperimentConfig
from src.model_adapter import monkey_patch_model, check_kv_consistency
from src.complete_experiment_runner import CompleteExperimentRunner


def load_gpt2_model(model_name: str = 'gpt2', device_name: str = 'auto'):
    """
    加载GPT-2模型
    
    Args:
        model_name: 模型名称
        device: 设备
        
    Returns:
        model, tokenizer, device
    """
    print(f"Loading {model_name}...")
    
    # 加载模型和tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 移动到设备
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    model = model.to(device) # type: ignore
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Model config: {model.config}")
    
    return model, tokenizer, device

class HFDatasetWrapper(TorchDataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return self.ds[idx]
    
def prepare_dataset(tokenizer, dataset_name: str = 'wikitext', 
                   subset: str = 'wikitext-2-raw-v1',
                   split: str = 'test',
                   max_samples: int = 100,
                   seq_length: int = 128,
                   batch_size: int = 4):
    """
    准备数据集
    
    Args:
        tokenizer: tokenizer
        dataset_name: 数据集名称
        subset: 子集名称
        split: 数据集分割
        max_samples: 最大样本数
        seq_length: 序列长度
        batch_size: batch大小
        
    Returns:
        dataloader
    """
    print(f"Loading dataset {dataset_name}/{subset}...")
    
    # 加载数据集
    dataset = load_dataset(dataset_name, subset, split=split)
    
    # 处理数据
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=seq_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    assert isinstance(dataset, HFDataset)
    
    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 限制样本数
    if max_samples and len(tokenized) > max_samples:
        tokenized = tokenized.select(range(max_samples))
    
    # 创建DataLoader
    tokenized.set_format('torch')
    
    wrapped = HFDatasetWrapper(tokenized)
    dataloader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"Dataset loaded: {len(tokenized)} samples")
    
    return dataloader


def check_model_kv(model, model_type: str = 'gpt2'):
    """检查模型的KV权重一致性"""
    print("\n" + "="*60)
    print("Checking K-V weight consistency...")
    print("="*60)
    
    kv_check = check_kv_consistency(model, model_type)
    
    print(f"Model type: {kv_check['model_type']}")
    print(f"Overall K=V: {kv_check['kv_equal']}")
    
    for layer_info in kv_check['layer_info'][:3]:  # 显示前3层
        print(f"  Layer {layer_info['layer_idx']}: K=V? {layer_info['kv_equal']}")
    
    if not kv_check['kv_equal']:
        print("\n⚠️  Warning: K and V weights are NOT equal!")
        print("   This may result in looser bounds.")
        print("   Options:")
        print("   1. Continue with current model (use relaxed bounds)")
        print("   2. Force K=V by weight sharing (experimental)")
        print("   3. Train a new model with K=V constraint")
    
    print("="*60 + "\n")
    
    return kv_check


def run_simple_experiment():
    """运行简单实验"""
    
    print("\n" + "="*60)
    print("Simple Experiment - GPT-2 with Single Bit Injection")
    print("="*60 + "\n")
    
    # 1. 配置
    config = ExperimentConfig(
        exp_name="gpt2_simple_test",
        model_name="gpt2",
        dataset_name="wikitext",
        batch_size=2,
        seq_length=64,
        num_samples=10,
        num_runs=1,
        
        # 注入配置
        injection_enabled=True,
        injection_location="scores",
        injection_layers=[0],  # 只在第0层注入
        injection_idx=(0, 0, 2, 7),
        injection_bit=15,
        
        seed=42
    )
    
    # 保存配置
    config.save()
    print(f"✓ Config saved to {config.save_dir}/{config.exp_id}/config.json\n")
    
    # 2. 加载模型
    model, tokenizer, device = load_gpt2_model('gpt2', 'auto')
    
    # 3. Monkey patch模型
    print("Patching model...")
    model = monkey_patch_model(
        model, 
        'gpt2', 
        injection_layers=config.injection_layers,
        force_kv_equal=False  # 不强制K=V
    )
    print("✓ Model patched\n")
    
    # 4. 检查KV一致性
    kv_check = check_model_kv(model, 'gpt2')
    
    # 5. 准备数据
    dataloader = prepare_dataset(
        tokenizer,
        dataset_name='wikitext',
        subset='wikitext-2-raw-v1',
        max_samples=config.num_samples,
        seq_length=config.seq_length,
        batch_size=config.batch_size
    )
    
    # 6. 运行实验
    runner = CompleteExperimentRunner(config)
    results = runner.run_all(model, dataloader)
    
    # 7. 打印结果
    print("\n" + "="*60)
    print("Experiment Results")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\nRun {i}:")
        print(f"  Baseline loss:    {result.baseline_loss:.6f}")
        print(f"  Injected loss:    {result.injected_loss:.6f}")
        print(f"  Loss diff:        {result.loss_diff:.6f}")
        print(f"  Violation detected: {result.violation_detected}")
        print(f"  Num violations:   {result.num_violations}")
        
        # 新增：打印epsilon分析
        if 'epsilon_analysis' in result.extra_data:
            print(f"\n  Epsilon Analysis:")
            for layer_analysis in result.extra_data['epsilon_analysis']:
                layer_idx = layer_analysis['layer_idx']
                analysis = layer_analysis['analysis']
                
                print(f"\n    Layer {layer_idx}:")
                print(f"      Mean epsilon diff:    {analysis['mean_diff']:.6f}")
                print(f"      Std epsilon diff:     {analysis['std_diff']:.6f}")
                print(f"      Max |epsilon diff|:   {analysis['max_abs_diff']:.6f}")
                
                print(f"\n      Top 5 Epsilon Changes:")
                for j, (pos, changes, bounds) in enumerate(analysis['top_changes'][:5]):
                    print(f"        #{j+1} Position {pos}:")
                    print(f"          Baseline ε:  {changes['baseline_epsilon']:.6f}")
                    print(f"          Injected ε:  {changes['injected_epsilon']:.6f}")
                    print(f"          Δε:          {changes['epsilon_diff']:.6f}")
                    print(f"          Bounds: [{bounds['middle']:.6f}, {bounds['upper']:.6f}]")
                    print(f"          γ (margin):  {bounds['gamma']:.6f}")
    
    print("\n" + "="*60)
    print(f"✓ Experiment completed!")
    print(f"✓ Results saved to {config.save_dir}/{config.exp_id}/")
    print("="*60 + "\n")

def run_parameter_sweep_experiment():
    """运行参数扫描实验 - 五维参数组合"""
    
    print("\n" + "="*60)
    print("Parameter Sweep - Comprehensive Multi-dimensional Scan")
    print("="*60 + "\n")
    
    from src.experiment_config import ParameterSweepConfig
    
    # 基础配置
    base_config = ExperimentConfig(
        exp_name="gpt2_comprehensive_sweep",
        model_name="gpt2",
        batch_size=2,
        seq_length=64,
        num_samples=10,  # 每个配置用较少样本
        num_runs=1,      # 每个配置运行1次
        
        injection_enabled=True,
        seed=42,
        
        bound_type="comb", # s@w, q@o, comb
        
        save_dir="./results/sweep_experiments_k=v_comb"
    )
    
    # 五维参数扫描配置
    sweep_config = ParameterSweepConfig(
        base_config=base_config,
        sweep_params={
            # 1. 随机种子
            # 'seed': [0, 42],
            'seed': [0, 42, 123, 3407],
            
            # 2. 注入层
            'injection_layers': [
                [0],        # 只在第0层
                [3],        # 只在第1层
                [6],
                [9],
            ],
            
            # 3. 比特位
            # 'injection_bit': [0, 7, 15, 23, 31],  # 从低到高
            'injection_bit': [b for b in range(0, 32)], 
            
            # 4. 注意力计算过程（张量类型）
            'injection_location': [
                'scores',   # attention scores (QK^T/√d)
                'weights',  # attention weights (softmax后)
                'q',        # query
                'k',        # key
                # 'v',        # value
                # 'none'
            ],
            
            # 5. 张量注入位置（空间位置）
            'injection_idx': [
                (0, 0, 10, 20),
                (0, 3, 53, 43),
                (1, 6, 32, 1),
                (1, 9, 31, 62),
            ],
        }
    )
    
    total_configs = sweep_config.get_num_configs()
    print(f"Total configurations: {total_configs}")
    print(f"= 4 seeds × 4 layer configs × 32 bits × 4 locations × 4 positions")
    print(f"= {total_configs} configs\n")
    
    from src.experiment_runner import ViolationLogger
    # ===== 创建违背日志记录器 =====
    violation_logger = ViolationLogger(
        save_dir=base_config.save_dir + "/violations",
        exp_name=base_config.exp_name
    )
    print(f"✓ Violation logger initialized")
    print(f"  Log file: {violation_logger.log_file}\n")
    
    # 加载模型(只加载一次)
    print("Loading model (once for all configs)...")
    model_base, tokenizer, device = load_gpt2_model('gpt2', 'auto')
    print("✓ Model loaded\n")
    
    # 准备数据(只准备一次)
    print("Preparing dataset...")
    dataloader = prepare_dataset(
        tokenizer,
        dataset_name='wikitext',
        subset='wikitext-2-raw-v1',
        max_samples=base_config.num_samples,
        seq_length=base_config.seq_length,
        batch_size=base_config.batch_size
    )
    print("✓ Dataset prepared\n")
    
    # 运行所有配置
    all_results = []
    
    print("="*60)
    print("Starting Parameter Sweep")
    print("="*60 + "\n")
    
    for config_idx, config in enumerate(sweep_config.generate_configs(), 1):
        print("\n" + "="*60)
        print(f"Configuration {config_idx}/{total_configs}")
        print("="*60)
        print(f"  Seed:              {config.seed}")
        print(f"  Injection Layers:  {config.injection_layers}")
        print(f"  Bit Position:      {config.injection_bit}")
        print(f"  Tensor Type:       {config.injection_location}")
        print(f"  Spatial Position:  {config.injection_idx}")
        print("="*60 + "\n")
        
        # 为这个配置patch模型（创建副本）
        import copy
        model = copy.deepcopy(model_base)
        model = monkey_patch_model(
            model,
            'gpt2',
            injection_layers=config.injection_layers,
            force_kv_equal=True
        )
        
        # 运行实验
        runner = CompleteExperimentRunner(config)
        results = runner.run_all(model, dataloader)
        
        # 打印结果（和run_simple_experiment一致的格式）
        print("\n" + "-"*60)
        print(f"Results for Config {config_idx}")
        print("-"*60)
        
        for i, result in enumerate(results):
            print(f"\nRun {i}:")
            print(f"  Baseline loss:      {result.baseline_loss:.6f}")
            print(f"  Injected loss:      {result.injected_loss:.6f}")
            print(f"  Loss diff:          {result.loss_diff:.6f}")
            print(f"  Violation detected: {result.violation_detected}")
            print(f"  Num violations:     {result.num_violations}")
            
            if result.violation_detected:
                violation_logger.log_violation(config_idx, config, result)
                print(f"  ⚠️  Violation logged to file")
                
            # 打印epsilon分析
            if 'epsilon_analysis' in result.extra_data:
                print(f"\n  Epsilon Analysis:")
                for layer_analysis in result.extra_data['epsilon_analysis']:
                    layer_idx = layer_analysis['layer_idx']
                    analysis = layer_analysis['analysis']
                    
                    print(f"\n    Layer {layer_idx}:")
                    print(f"      Mean epsilon diff:    {analysis['mean_diff']:.6f}")
                    print(f"      Std epsilon diff:     {analysis['std_diff']:.6f}")
                    print(f"      Max |epsilon diff|:   {analysis['max_abs_diff']:.6f}")
                    
                    # 只打印top 3来节省空间
                    if len(analysis['top_changes']) > 0:
                        print(f"\n      Top 3 Epsilon Changes:")
                        for j, (pos, changes, bounds) in enumerate(analysis['top_changes'][:3]):
                            print(f"        #{j+1} Position {pos}:")
                            print(f"          Baseline ε:  {changes['baseline_epsilon']:.6f}")
                            print(f"          Injected ε:  {changes['injected_epsilon']:.6f}")
                            print(f"          Δε:          {changes['epsilon_diff']:.6f}")
                            print(f"          Bounds: [{bounds['middle']:.6f}, {bounds['upper']:.6f}]")
                            print(f"          γ (margin):  {bounds['gamma']:.6f}")
        
        all_results.extend(results)
        
        print("\n" + "-"*60)
        print(f"✓ Config {config_idx}/{total_configs} completed")
        print("-"*60 + "\n")
        
        # 清理模型以节省内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 最终汇总
    print("\n" + "="*60)
    print("Parameter Sweep Summary")
    print("="*60)
    print(f"Total configurations tested: {total_configs}")
    print(f"Total runs completed:        {len(all_results)}")
    
    # 统计违反情况
    num_violations = sum(1 for r in all_results if r.violation_detected)
    print(f"Configs with violations:     {num_violations} ({100*num_violations/len(all_results):.1f}%)")
    
    # 找出loss变化最大的配置
    if all_results:
        max_loss_diff_result = max(all_results, key=lambda r: abs(r.loss_diff) if r.loss_diff is not None else 0)
        print(f"\nLargest loss change:")
        print(f"  Config: {max_loss_diff_result.exp_id}")
        print(f"  Loss diff: {max_loss_diff_result.loss_diff:.6f}")
        if max_loss_diff_result.injection_info:
            print(f"  Location: {max_loss_diff_result.injection_info.get('location')}")
            print(f"  Bit: {max_loss_diff_result.injection_info.get('bit')}")
    
    print("\n" + "="*60)
    print("✓ Parameter sweep completed!")
    print(f"✓ All results saved to {base_config.save_dir}/")
    print("="*60 + "\n")
    
    return all_results

# def run_parameter_sweep_experiment():
#     """运行参数扫描实验"""
    
#     print("\n" + "="*60)
#     print("Parameter Sweep - Multiple Bit Positions")
#     print("="*60 + "\n")
    
#     from src.experiment_config import ParameterSweepConfig
    
#     # 基础配置
#     base_config = ExperimentConfig(
#         exp_name="gpt2_bit_sweep",
#         model_name="gpt2",
#         batch_size=2,
#         seq_length=64,
#         num_samples=20,
#         num_runs=2,
        
#         injection_enabled=True,
#         injection_location="scores",
#         injection_layers=[0, 1],  # 在前两层注入
#         injection_idx=(0, 0, 2, 7),
#     )
    
#     # 参数扫描
#     sweep_config = ParameterSweepConfig(
#         base_config=base_config,
#         sweep_params={
#             'injection_bit': [0, 7, 15, 23, 31],  # 5个比特位
#             'seed': [42, 123],  # 2个种子
#         }
#     )
    
#     print(f"Total configurations: {sweep_config.get_num_configs()}")
#     print(f"= 5 bits × 2 seeds = 10 configs\n")
    
#     # 加载模型(只加载一次)
#     model, tokenizer, device = load_gpt2_model('gpt2')
    
#     # 准备数据(只准备一次)
#     dataloader = prepare_dataset(
#         tokenizer,
#         max_samples=base_config.num_samples,
#         seq_length=base_config.seq_length,
#         batch_size=base_config.batch_size
#     )
    
#     # 运行所有配置
#     all_results = []
    
#     for i, config in enumerate(sweep_config.generate_configs()):
#         print(f"\n{'='*60}")
#         print(f"Configuration {i+1}/{sweep_config.get_num_configs()}")
#         print(f"Bit: {config.injection_bit}, Seed: {config.seed}")
#         print(f"{'='*60}")
        
#         # Patch模型
#         model_copy = monkey_patch_model(
#             model,
#             'gpt2',
#             injection_layers=config.injection_layers
#         )
        
#         # 运行实验
#         runner = CompleteExperimentRunner(config)
#         results = runner.run_all(model_copy, dataloader)
        
#         all_results.extend(results)
    
#     print(f"\n✓ Parameter sweep completed: {len(all_results)} total runs")


def run_multi_layer_experiment():
    """运行多层注入实验"""
    
    print("\n" + "="*60)
    print("Multi-Layer Injection Experiment")
    print("="*60 + "\n")
    
    config = ExperimentConfig(
        exp_name="gpt2_multi_layer",
        model_name="gpt2",
        batch_size=2,
        seq_length=64,
        num_samples=20,
        num_runs=3,
        
        injection_enabled=True,
        injection_location="scores",
        injection_layers=[0, 3, 6, 9],  # 在4个层注入
        injection_idx=(0, 0, 2, 7),
        injection_bit=15,
        
        seed=42
    )
    
    model, tokenizer, device = load_gpt2_model('gpt2')
    
    model = monkey_patch_model(
        model,
        'gpt2',
        injection_layers=config.injection_layers
    )
    
    dataloader = prepare_dataset(
        tokenizer,
        max_samples=config.num_samples,
        seq_length=config.seq_length,
        batch_size=config.batch_size
    )
    
    runner = CompleteExperimentRunner(config)
    results = runner.run_all(model, dataloader)
    
    print(f"\n✓ Multi-layer experiment completed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPU Silent Data Corruption Detection Framework")
    print("Complete Usage Examples")
    print("="*60)
    
    # 选择运行哪个示例
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        print("\nAvailable examples:")
        print("  1. simple       - Simple single run")
        print("  2. sweep        - Parameter sweep")
        print("  3. multilayer   - Multi-layer injection")
        print("  4. all          - Run all examples")
        print()
        example = input("Select example (1-4): ").strip()
    
    if example in ['1', 'simple']:
        run_simple_experiment()
    elif example in ['2', 'sweep']:
        run_parameter_sweep_experiment()
    elif example in ['3', 'multilayer']:
        run_multi_layer_experiment()
    elif example in ['4', 'all']:
        run_simple_experiment()
        run_parameter_sweep_experiment()
        run_multi_layer_experiment()
    else:
        print(f"Unknown example: {example}")
        print("Running simple example by default...")
        run_simple_experiment()