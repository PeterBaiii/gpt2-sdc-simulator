# gpt2-sdc-simulator

基于注意力机制理论边界的静默数据错误(SDC)检测框架,用于大规模GPU集群中LLM训练/推理的错误检测。

## 📁 模块结构

```
project/
├── utils/
│   ├── debug.py              # 调试工具
│   ├── check_nan.py          # NaN检测
│   ├── bound_fixing.py       # 边界检查
│   └── return_top2.py        # Top-2选择
│
├── fault_injection.py        # 错误注入模块 ⭐
├── bounds_computation.py     # 边界计算模块 ⭐
├── experiment_config.py      # 实验配置模块 ⭐
├── experiment_runner.py      # 实验运行模块 ⭐
├── model_adapter.py          # 模型适配器模块 ⭐
├── example_usage.py          # 使用示例
│
├── minimal_task.py           # 最小任务(原始实验)
└── README.md                 # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets scipy numpy
```

### 2. 单次实验

```python
from fault_injection import InjectionConfig, InjectionLocation
from experiment_config import ExperimentConfig
from experiment_runner import ExperimentRunner

# 创建配置
config = ExperimentConfig(
    exp_name="my_experiment",
    model_name="gpt2",
    injection_location="scores",
    injection_bit=15,
    seed=42
)

# 运行实验
runner = ExperimentRunner(config)
results = runner.run_all(model, dataloader)
```

### 3. 参数扫描

```python
from experiment_config import ConfigTemplates

# 使用预定义模板
sweep_config = ConfigTemplates.bit_sweep()

# 或自定义扫描
from experiment_config import ParameterSweepConfig

sweep = ParameterSweepConfig(
    base_config=base_config,
    sweep_params={
        'seed': [42, 123, 456],
        'injection_bit': list(range(32)),
        'injection_location': ['scores', 'weights']
    }
)

# 运行扫描
from experiment_runner import run_parameter_sweep
results = run_parameter_sweep(sweep, model_fn, data_fn)
```

## 📦 核心模块详解

### 1. fault_injection.py - 错误注入

**功能:**
- 单/多比特翻转
- 随机位置注入
- 概率性注入
- 注入历史记录

**主要类:**
- `InjectionConfig`: 注入配置
- `FaultInjector`: 注入器类

**示例:**
```python
from fault_injection import InjectionConfig, InjectionLocation, FaultInjector

config = InjectionConfig(
    location=InjectionLocation.SCORES,
    idx=(0, 0, 2, 7),
    bit=15,
    enabled=True
)

injector = FaultInjector(config)
info = injector.inject(tensor)
```

### 2. bounds_computation.py - 边界计算

**功能:**
- 计算注意力机制理论上下界
- 基于Lambert W函数的紧致边界
- NaN/Inf处理
- 违反检测

**主要函数:**
- `compute_attention_bounds()`: 计算边界
- `detect_violation()`: 检测违反
- `compute_injected_epsilon()`: 计算注入后的epsilon

**示例:**
```python
from bounds_computation import compute_attention_bounds, detect_violation

bounds = compute_attention_bounds(scores, p, d=16)
result = detect_violation(bounds, injected_epsilon)
```

### 3. experiment_config.py - 实验配置

**功能:**
- 统一的配置管理
- 参数扫描生成
- 配置保存/加载
- 预定义模板

**主要类:**
- `ExperimentConfig`: 实验配置
- `ParameterSweepConfig`: 参数扫描配置
- `ConfigTemplates`: 预定义模板

### 4. experiment_runner.py - 实验运行

**功能:**
- 实验执行
- 结果记录
- 日志管理
- 汇总统计

**主要类:**
- `ExperimentRunner`: 实验运行器
- `ResultsLogger`: 结果记录器
- `ExperimentResult`: 结果数据类

### 5. model_adapter.py - 模型适配

**功能:**
- 统一的注入接口
- 中间张量捕获
- Monkey patching
- 多模型支持

**支持的模型:**
- ✅ GPT-2
- ✅ DistilBERT
- 🚧 TinyLlama
- 🚧 OPT

## 🎯 推荐的开源模型

### 小规模模型 (< 200M参数)

| 模型 | 参数量 | 推荐理由 | HuggingFace ID |
|------|--------|----------|----------------|
| **GPT-2 Small** | 124M | 代码成熟,易于hack | `gpt2` |
| **DistilBERT** | 66M | 更小更快,双向注意力 | `distilbert-base-uncased` |
| **OPT-125M** | 125M | 类GPT架构 | `facebook/opt-125m` |
| **Pythia-70M** | 70M | 多检查点,适合分析 | `EleutherAI/pythia-70m` |
| **Pythia-160M** | 160M | 同上 | `EleutherAI/pythia-160m` |

### 中等规模模型 (200M-1B参数)

| 模型 | 参数量 | 推荐理由 | HuggingFace ID |
|------|--------|----------|----------------|
| **GPT-2 Medium** | 355M | 平衡性能和规模 | `gpt2-medium` |
| **OPT-350M** | 350M | 类GPT架构 | `facebook/opt-350m` |
| **Pythia-410M** | 410M | 多检查点 | `EleutherAI/pythia-410m` |
| **TinyLlama** | 1.1B | 现代架构,高效 | `TinyLlama/TinyLlama-1.1B` |

### 推荐选择 (优先级排序)

1. **GPT-2 (首选)** ⭐⭐⭐
   - 最成熟的实现
   - 丰富的社区资源
   - 容易hack
   
2. **DistilBERT** ⭐⭐⭐
   - 最小模型,快速实验
   - 双向注意力,测试不同场景
   
3. **Pythia系列** ⭐⭐
   - 多个训练检查点
   - 适合研究训练过程

4. **TinyLlama** ⭐⭐
   - 现代架构(LLaMA)
   - 1B参数仍可管理

## 📊 推荐的数据集

### 语言建模数据集

| 数据集 | 大小 | 推荐理由 | HuggingFace ID |
|--------|------|----------|----------------|
| **WikiText-2** | 2M tokens | 小巧,快速测试 | `wikitext-2-raw-v1` |
| **WikiText-103** | 100M tokens | 中等规模 | `wikitext-103-raw-v1` |
| **OpenWebText** | 8M docs | GPT-2训练集 | `openwebtext` |
| **C4 (subset)** | 可定制 | 大规模语料 | `c4` |

### 下游任务数据集

| 数据集 | 任务类型 | HuggingFace ID |
|--------|----------|----------------|
| **GLUE** | 多任务benchmark | `glue` |
| **LAMBADA** | 语言理解 | `lambada` |
| **HellaSwag** | 常识推理 | `hellaswag` |

### 推荐选择

1. **WikiText-2** (首选,快速原型) ⭐⭐⭐
2. **WikiText-103** (中等规模实验) ⭐⭐⭐
3. **OpenWebText** (更真实场景) ⭐⭐
4. **C4 subset** (大规模测试) ⭐

## 🔧 实验流程

### 标准流程

```
1. 配置 → 2. 加载模型/数据 → 3. Monkey patch → 4. Baseline运行 
→ 5. 注入运行 → 6. 计算边界 → 7. 检测违反 → 8. 保存结果
```

### 参数扫描流程

```
1. 定义扫描范围 → 2. 生成所有配置 → 3. 批量运行 → 4. 汇总分析
```

## 📈 实验设计建议

### 初步实验 (快速验证)

```python
# 小模型 + 小数据 + 少量扫描
config = ExperimentConfig(
    model_name="distilbert-base-uncased",
    dataset_name="wikitext-2-raw-v1",
    batch_size=4,
    num_samples=20,
    num_runs=3
)

sweep_params = {
    'injection_bit': [0, 7, 15, 23, 31],  # 5个比特
    'injection_location': ['scores', 'weights'],  # 2个位置
    'seed': [42, 123]  # 2个种子
}
# 总配置数: 5 × 2 × 2 = 20
```

### 中等规模实验

```python
# GPT-2 + WikiText-103 + 中等扫描
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="wikitext-103-raw-v1",
    batch_size=8,
    num_samples=100,
    num_runs=5
)

sweep_params = {
    'injection_bit': list(range(32)),  # 32个比特
    'injection_location': ['q', 'k', 'v', 'scores', 'weights', 'out'],  # 6个位置
    'seed': [42, 123, 456, 789]  # 4个种子
}
# 总配置数: 32 × 6 × 4 = 768
```

### 大规模实验

```python
# 多模型 + 多数据集 + 完整扫描
models = ['gpt2', 'distilbert-base-uncased', 'EleutherAI/pythia-160m']
datasets = ['wikitext-103-raw-v1', 'openwebtext']

# 每个模型-数据集组合运行完整扫描
# 预计配置数: 3 × 2 × 768 = 4608
```

## 🎨 可视化分析

建议的分析维度:

1. **检测率分析**
   - 不同比特位的检测率
   - 不同注入位置的检测率
   - 不同模型的检测率对比

2. **边界紧致性分析**
   - epsilon分布
   - 违反margin统计
   - 上下界gap分析

3. **错误传播分析**
   - 不同层的影响
   - Loss变化vs违反检测
   - 时序传播模式

## 🔍 调试技巧

### 1. 启用详细日志

```python
from utils.debug import enable_debug, enable_log_file

enable_debug(True)
enable_log_file("debug.log")
```

### 2. 检查中间张量

```python
from utils.check_nan import check_nan

check_nan(scores, name="attention_scores")
check_nan(bounds.epsilon, name="epsilon")
```

### 3. 验证边界

```python
from utils.bound_fixing import hist_tensor_diff

hist_tensor_diff(bounds.to_dict())
```

## 📝 TODO

- [ ] 完善model_adapter中TinyLlama和OPT的支持
- [ ] 实现训练过程中的梯度注入
- [ ] 添加因果掩码的特殊处理
- [ ] 优化大规模实验的内存使用
- [ ] 添加可视化工具
- [ ] 支持分布式实验
- [ ] 添加更多预定义实验模板

## 📚 参考文献

详见 `Logs.md` 中的理论推导和复杂度分析。

## 🤝 贡献

欢迎贡献代码、报告bug或提出改进建议!

## 📄 License

[待定]