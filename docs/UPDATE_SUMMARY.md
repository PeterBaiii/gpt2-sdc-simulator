# 框架更新总结

根据你的要求，我已经完成了以下4项修正和补充：

## ✅ 1. 补足 ExperimentRunner 的 run_single 函数

### 更新内容

- **创建了 `complete_experiment_runner.py`** - 完整的实验运行器
- **实现了完整的实验流程**：
  1. Baseline运行（无错误注入）
  2. Injection运行（带错误注入）
  3. 边界计算
  4. 违反检测
  5. 性能监控

### 核心组件

```python
class CompleteExperimentRunner:
    def run_single(self, run_id, model, dataloader):
        # 1. 运行baseline
        baseline_outputs, baseline_tensors = run_model_with_collection(...)
        
        # 2. 运行injection
        injected_outputs, injected_tensors = run_model_with_collection(...)
        
        # 3. 计算边界
        bounds = compute_attention_bounds(scores, weights, d)
        
        # 4. 检测违反
        detection = detect_violation(bounds, injected_eps, tolerance)
        
        return result
```

### 关键特性

- ✅ 自动收集中间张量 (q, k, v, scores, weights)
- ✅ 支持多层处理
- ✅ 完整的性能监控
- ✅ 结果自动保存

---

## ✅ 2. 支持多层注入配置

### 更新内容

#### 2.1 配置层面 (`experiment_config.py`)

添加了新的配置参数：

```python
@dataclass
class ExperimentConfig:
    # 层选择配置
    injection_layers: Optional[List[int]] = None  # 要注入的层索引
    injection_layer_mode: str = "first"  # first, all, random, specific
```

#### 2.2 模型适配层面 (`model_adapter.py`)

更新了 `monkey_patch_model` 函数：

```python
def monkey_patch_model(model, model_type: str, 
                       injection_layers: Optional[List[int]] = None,
                       force_kv_equal: bool = False):
    """
    Args:
        injection_layers: 要注入的层索引列表
            - None: 所有层
            - [0, 1, 2]: 只在这些层注入
    """
```

### 使用示例

```python
# 只在第0层注入
config.injection_layers = [0]

# 在前3层注入
config.injection_layers = [0, 1, 2]

# 在特定层注入
config.injection_layers = [0, 3, 6, 9]

# 所有层注入
config.injection_layers = None
```

---

## ✅ 3. 添加性能检测模块

### 新增文件：`performance_monitor.py`

#### 3.1 性能指标类

```python
@dataclass
class PerformanceMetrics:
    # 时间指标
    total_time: float
    baseline_forward_time: float
    injection_forward_time: float
    bounds_computation_time: float
    violation_detection_time: float
    
    # 内存指标
    peak_memory_allocated: float  # MB
    peak_memory_reserved: float
    baseline_memory: float
    injection_memory: float
    
    # 计算指标
    num_attention_layers: int
    num_attention_heads: int
    sequence_length: int
    batch_size: int
    attention_flops: float
    bounds_flops: float
```

#### 3.2 性能监控器

```python
class PerformanceMonitor:
    # 计时器
    with monitor.timer('baseline_forward'):
        ...
    
    # 内存记录
    monitor.record_memory('baseline')
    
    # 模型信息
    monitor.record_model_info(model, batch_size, seq_length)
    
    # FLOPS估算
    monitor.estimate_flops(d_model, n_heads, seq_len, batch_size)
```

#### 3.3 性能聚合器

用于多次实验的统计分析：

```python
class PerformanceAggregator:
    def add(self, metrics)
    def compute_statistics()  # mean, std, min, max, median
    def print_statistics()
```

### 监控的指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **时间** | baseline_forward_time | Baseline前向传播时间 |
| | injection_forward_time | 注入版前向传播时间 |
| | bounds_computation_time | 边界计算时间 |
| | violation_detection_time | 违反检测时间 |
| | total_time | 总时间 |
| **内存** | peak_memory_allocated | 峰值已分配内存 (MB) |
| | peak_memory_reserved | 峰值预留内存 (MB) |
| | baseline_memory | Baseline内存使用 |
| | injection_memory | 注入版内存使用 |
| **开销** | injection_vs_baseline | 注入vs基线的时间开销 (%) |
| | detection_vs_baseline | 检测vs基线的时间开销 (%) |
| | total_vs_baseline | 总开销 (%) |

### 输出示例

```
==============================================================
Performance Summary
==============================================================

[Time Metrics]
  Total time:              2.3456s
  Baseline forward:        0.8234s
  Injection forward:       0.8567s
  Bounds computation:      0.4123s
  Violation detection:     0.1234s

[Overhead]
  injection_vs_baseline: +4.04%
  detection_vs_baseline: +65.09%
  total_vs_baseline: +69.13%

[Memory Metrics]
  Peak allocated:          512.34 MB
  Peak reserved:           1024.00 MB
  Baseline memory:         487.23 MB
  Injection memory:        489.12 MB

[Model Info]
  Attention layers:        12
  Attention heads:         12
  Sequence length:         128
  Batch size:              4

[FLOPS Estimate]
  Attention:               2.51e+09
  Bounds computation:      6.29e+06
  Effective TFLOPS:        3.05
==============================================================
```

---

## ✅ 4. KV权重一致性处理

### 4.1 检查KV一致性

新增函数 `check_kv_consistency()`:

```python
def check_kv_consistency(model, model_type: str) -> Dict:
    """
    检查模型的K和V权重是否一致
    
    Returns:
        {
            'kv_equal': bool,  # 是否所有层K=V
            'layer_info': [
                {
                    'layer_idx': int,
                    'kv_equal': bool,
                    'k_shape': list,
                    'v_shape': list
                },
                ...
            ]
        }
    """
```

### 4.2 使用示例

```python
# 1. 加载预训练GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 2. 检查KV一致性
kv_check = check_kv_consistency(model, 'gpt2')

if kv_check['kv_equal']:
    print("✓ K and V are equal - can use tight bounds")
else:
    print("⚠️  K and V are NOT equal")
    print("Options:")
    print("1. Use relaxed bounds (current)")
    print("2. Force K=V by weight sharing (experimental)")
    print("3. Train new model with K=V constraint")
```

### 4.3 关于预训练GPT-2的KV权重

**实际情况：**
- ❌ GPT-2的预训练模型 **K ≠ V**
- ✅ GPT-2使用独立的QKV线性层
- ⚠️  理论边界可能不够紧致

**解决方案：**

#### 选项1：使用当前框架（推荐）✅

```python
# 直接使用，不强制K=V
model = monkey_patch_model(
    model, 
    'gpt2', 
    force_kv_equal=False  # 使用放松的边界
)
```

**优点：**
- 立即可用，无需重新训练
- 适合快速原型和实验
- 仍然可以检测到大部分错误

**缺点：**
- 边界可能较松
- 检测灵敏度可能降低

#### 选项2：强制K=V（实验性）⚠️

```python
model = monkey_patch_model(
    model, 
    'gpt2', 
    force_kv_equal=True  # 强制共享权重
)
```

**注意：** 此功能尚未完全实现，需要：
- 修改模型权重使K和V共享
- 可能影响模型性能
- 需要额外的测试

#### 选项3：重新训练（最优但耗时）🎯

训练一个K=V的模型：

```python
# 自定义GPT-2配置
config = GPT2Config(...)

# 修改attention层使K=V
class CustomGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_q = nn.Linear(...)
        self.c_kv = nn.Linear(...)  # K和V共享
```

---

## 📁 新增/修改的文件

### 新增文件

1. ✨ **performance_monitor.py** - 性能检测模块
2. ✨ **complete_experiment_runner.py** - 完整实验运行器
3. ✨ **full_example.py** - 完整使用示例
4. 📝 **UPDATE_SUMMARY.md** - 本文档

### 修改文件

1. 🔧 **experiment_config.py** - 添加层选择配置
2. 🔧 **model_adapter.py** - 添加多层支持和KV检查
3. 🔧 **experiment_runner.py** - 更新run_single框架

---

## 🚀 快速开始

### 最简单的例子

```python
# 1. 安装依赖
pip install torch transformers datasets scipy numpy psutil

# 2. 运行简单实验
python full_example.py simple

# 3. 查看结果
ls results/gpt2_simple_test_*/
```

### 完整实验流程

```python
from experiment_config import ExperimentConfig
from model_adapter import monkey_patch_model, check_kv_consistency
from complete_experiment_runner import CompleteExperimentRunner
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. 配置
config = ExperimentConfig(
    exp_name="my_exp",
    injection_layers=[0, 1, 2],  # 前3层
    injection_location="scores",
    injection_bit=15
)

# 2. 加载模型
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 3. 检查KV
kv_check = check_kv_consistency(model, 'gpt2')
print(f"K=V? {kv_check['kv_equal']}")

# 4. Patch模型
model = monkey_patch_model(
    model, 
    'gpt2',
    injection_layers=config.injection_layers
)

# 5. 准备数据
# ... (见full_example.py)

# 6. 运行实验
runner = CompleteExperimentRunner(config)
results = runner.run_all(model, dataloader)

# 7. 查看结果
for result in results:
    print(f"Loss diff: {result.loss_diff}")
    print(f"Violations: {result.num_violations}")
```

---

## 📊 实验建议

### 对于KV不一致的预训练模型

#### 实验设置

```python
config = ExperimentConfig(
    # 使用较大的tolerance
    tolerance=1e-5,  # 而不是1e-6
    
    # 多次运行以获得统计
    num_runs=10,
    
    # 扫描多个比特位
    # (某些比特位可能更容易检测)
)

sweep_params = {
    'injection_bit': list(range(32)),
    'injection_location': ['scores', 'weights', 'out'],
    'seed': [42, 123, 456, 789]
}
```

#### 分析重点

1. **检测率分析**
   - 哪些比特位检测率高？
   - 哪些注入位置更容易检测？

2. **边界紧致度**
   - epsilon的分布
   - 违反margin的统计

3. **性能开销**
   - 检测时间 vs 基线时间
   - 内存开销

---

## ⚠️  已知限制

1. **KV不一致** - 预训练GPT-2的K≠V，边界可能较松
2. **中间张量收集** - 需要额外内存，大模型可能OOM
3. **多层注入** - 同时注入多层时，错误可能传播和叠加
4. **因果掩码** - 当前对掩码的处理可能不够完善

---

## 🔜 后续工作

### 短期（1周内）

- [ ] 完善因果掩码的边界处理
- [ ] 优化中间张量收集的内存使用
- [ ] 添加更多模型支持 (DistilBERT, OPT)

### 中期（2-4周）

- [ ] 实现训练过程中的梯度注入
- [ ] 支持强制K=V的权重共享
- [ ] 添加可视化工具

### 长期（1-3月）

- [ ] 训练一个K=V的模型
- [ ] 大规模参数扫描
- [ ] 优化边界公式（引入修正因子）

---

## 📚 文档

- `README.md` - 框架总体文档
- `UPDATE_SUMMARY.md` - 本文档
- `full_example.py` - 完整示例代码
- `Logs.md` - 理论推导

---

## 🎯 建议的实验顺序

```
1. 运行 full_example.py simple
   ↓
2. 检查结果，验证框架正常工作
   ↓
3. 运行小规模参数扫描 (5 bits × 2 seeds)
   ↓
4. 分析初步结果
   ↓
5. 运行中等规模扫描 (32 bits × 6 locations × 4 seeds)
   ↓
6. 根据结果调整tolerance和配置
   ↓
7. 运行大规模实验
```

---

现在你有一个**完整、可运行、支持多层注入和性能监控**的框架了！🎉