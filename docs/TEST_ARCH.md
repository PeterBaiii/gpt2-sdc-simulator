# test/ 目录架构说明

## 概述

`test/` 目录包含实验运行脚本、结果分析和可视化工具。核心是 `run_experiment.py`，提供了完整的实验运行流程。

## 文件列表

### 1. run_experiment.py - 主实验脚本 ⭐

**功能**：完整的实验运行脚本，从模型加载到结果保存的端到端流程。

#### 运行方式

##### 方法1: 命令行参数

```bash
cd test

# 运行参数扫描（推荐，用于复现实验）
python run_experiment.py sweep

# 运行简单单次实验
python run_experiment.py simple

# 运行多层注入实验
python run_experiment.py multilayer

# 运行所有实验
python run_experiment.py all
```

##### 方法2: 交互式选择

```bash
cd test
python run_experiment.py

# 将显示菜单：
# Available examples:
#   1. simple       - Simple single run
#   2. sweep        - Parameter sweep ⭐
#   3. multilayer   - Multi-layer injection
#   4. all          - Run all examples
```

#### 核心函数

##### run_simple_experiment()
```python
def run_simple_experiment():
    """运行简单单次实验"""
```

**功能**：
- 单一配置的基础实验
- 适合快速验证和调试
- 完整显示epsilon分析

**配置示例**：
```python
config = ExperimentConfig(
    exp_name="gpt2_simple_test",
    model_name="gpt2",
    batch_size=2,
    seq_length=64,
    num_samples=10,
    
    # 注入配置
    injection_enabled=True,
    injection_location="scores",
    injection_layers=[0],
    injection_idx=(0, 0, 2, 7),
    injection_bit=15,
    
    # ⭐ 边界配置
    bound_type="s@w",  # 可修改
    
    seed=42
)

# 加载并patch模型
model, tokenizer, device = load_gpt2_model('gpt2')
model = monkey_patch_model(
    model, 'gpt2',
    injection_layers=config.injection_layers,
    force_kv_equal=False  # ⭐ 可修改
)
```

**输出示例**：
```
Run 0:
  Baseline loss:    3.456789
  Injected loss:    3.456821
  Loss diff:        0.000032
  Violation detected: True
  Num violations:   15

  Epsilon Analysis:
    Layer 0:
      Mean epsilon diff:    0.001234
      Std epsilon diff:     0.000567
      Max |epsilon diff|:   0.005678
      
      Top 5 Epsilon Changes:
        #1 Position (0, 0, 10, 20):
          Baseline ε:  2.345678
          Injected ε:  2.351356
          Δε:          0.005678
          Bounds: [1.234567, 3.456789]
          γ (margin):  0.987654
```

##### run_parameter_sweep_experiment() ⭐ 推荐

```python
def run_parameter_sweep_experiment():
    """运行参数扫描实验 - 用于复现论文结果"""
```

**功能**：
- 系统性扫描多个参数组合
- 自动记录所有违反
- 生成完整实验报告
- **这是复现实验结果的主要方法**

**扫描配置**：

```python
# 基础配置
base_config = ExperimentConfig(
    exp_name="gpt2_sweep",
    model_name="gpt2",
    batch_size=4,
    seq_length=128,
    num_samples=100,
    num_runs=1,
    
    # ⭐ 边界计算方法
    bound_type="comb",  # "s@w" / "q@o" / "comb"
    bounds_check=True,
    tolerance=0.0,
    
    seed=42
)

# 扫描参数
sweep_config = ParameterSweepConfig(
    base_config=base_config,
    sweep_params={
        # 1. 随机种子（4个）
        'seed': [0, 42, 123, 3407],
        
        # 2. 注入层（4个配置）
        'injection_layers': [[0], [3], [6], [9]],
        
        # 3. 比特位（32个）
        'injection_bit': list(range(32)),
        
        # 4. 张量类型（4个）
        'injection_location': ['scores', 'weights', 'q', 'k'],
        
        # 5. 空间位置（4个）
        'injection_idx': [
            (0, 0, 10, 20),
            (0, 3, 53, 43),
            (1, 6, 32, 1),
            (1, 9, 31, 62),
        ],
    }
)

# 总配置数: 4 × 4 × 32 × 4 × 4 = 8192
```

**关键参数修改位置** ⭐：

1. **修改 bound_type**：
```python
# 第258行附近
base_config = ExperimentConfig(
    bound_type="comb",  # ⭐ 修改此处
    # "s@w"   - scores × weights 路径
    # "q@o"   - query × output 路径
    # "comb"  - 组合两种路径（推荐）
)
```

2. **修改 force_kv_equal**：
```python
# 第368行附近
model = monkey_patch_model(
    model, 'gpt2',
    injection_layers=config.injection_layers,
    force_kv_equal=True  # ⭐ 修改此处
    # False - 不强制K=V（使用原始权重）
    # True  - 强制K=V（推荐，检测率更高）
)
```

3. **修改扫描参数范围**：
```python
# 第285-312行
sweep_params={
    'seed': [0, 42, 123, 3407],  # 可增减种子
    'injection_bit': list(range(32)),  # 可限制为高位比特
    # 例如只测试高位比特：
    # 'injection_bit': [23, 24, 25, 26, 27, 28, 29, 30, 31],
}
```

**执行流程**：

```python
# 1. 加载模型（只加载一次）
model_base, tokenizer, device = load_gpt2_model('gpt2')

# 2. 准备数据（只准备一次）
dataloader = prepare_dataset(tokenizer, ...)

# 3. 遍历所有配置
for config_idx, config in enumerate(sweep_config.generate_configs()):
    # 3.1 为当前配置创建模型副本
    model = copy.deepcopy(model_base)
    
    # 3.2 Patch模型（应用force_kv_equal）
    model = monkey_patch_model(
        model, 'gpt2',
        injection_layers=config.injection_layers,
        force_kv_equal=True  # ⭐
    )
    
    # 3.3 运行实验
    runner = CompleteExperimentRunner(config)
    results = runner.run_all(model, dataloader)
    
    # 3.4 记录违反
    for result in results:
        if result.violation_detected:
            violation_logger.log_violation(config_idx, config, result)
    
    # 3.5 清理模型
    del model
    torch.cuda.empty_cache()
```

**输出格式**：

```
Configuration 1/8192
============================================================
  Seed:              0
  Injection Layers:  [0]
  Bit Position:      0
  Tensor Type:       scores
  Spatial Position:  (0, 0, 10, 20)
============================================================

Run 0:
  Baseline loss:      3.456789
  Injected loss:      3.456789
  Loss diff:          0.000000
  Violation detected: False
  Num violations:     0

------------------------------------------------------------
✓ Config 1/8192 completed
------------------------------------------------------------

...

============================================================
Parameter Sweep Summary
============================================================
Total configurations tested: 8192
Total runs completed:        8192
Configs with violations:     596 (7.3%)

Largest loss change:
  Config: gpt2_sweep_seed=42_injection_bit=30_...
  Loss diff: 0.123456
  Location: weights
  Bit: 30

============================================================
✓ Parameter sweep completed!
✓ All results saved to ./results/
============================================================
```

**结果文件**：
```
results/
├── gpt2_sweep_20251219_103000/
│   ├── config.json
│   ├── logs/
│   │   └── experiment.log
│   └── violations/
│       ├── violation_log.txt      # 所有违反的配置
│       └── violation_summary.json # 违反统计
```

##### run_multi_layer_experiment()
```python
def run_multi_layer_experiment():
    """运行多层注入实验"""
```

**功能**：
- 同时在多个层注入
- 研究跨层错误传播

**配置**：
```python
config = ExperimentConfig(
    injection_layers=[0, 3, 6, 9],  # 4个层
    # 其他配置...
)
```

#### 辅助函数

##### load_gpt2_model()
```python
def load_gpt2_model(
    model_name: str = 'gpt2',
    device_name: str = 'auto'
) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer, torch.device]:
    """加载GPT-2模型和tokenizer"""
```

**功能**：
- 加载预训练模型
- 设置pad_token
- 自动选择设备（GPU/CPU）

##### prepare_dataset()
```python
def prepare_dataset(
    tokenizer,
    dataset_name: str = 'wikitext',
    subset: str = 'wikitext-2-raw-v1',
    max_samples: int = 100,
    seq_length: int = 128,
    batch_size: int = 4
) -> DataLoader:
    """准备WikiText数据集"""
```

**功能**：
- 加载HuggingFace数据集
- Tokenize和padding
- 创建DataLoader

##### check_model_kv()
```python
def check_model_kv(model, model_type: str = 'gpt2'):
    """检查模型的KV权重一致性"""
```

**输出**：
```
============================================================
Checking K-V weight consistency...
============================================================
Model type: gpt2
Overall K=V: False

  Layer 0: K=V? False
  Layer 1: K=V? False
  Layer 2: K=V? False

⚠️  Warning: K and V weights are NOT equal!
   This may result in looser bounds.
   Options:
   1. Continue with current model (use relaxed bounds)
   2. Force K=V by weight sharing (experimental)
   3. Train a new model with K=V constraint
============================================================
```

---

### 2. analyzer.py - 结果分析

**功能**：分析实验结果，生成统计报告。

#### 核心类

##### ExperimentAnalyzer
```python
class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_df = None
        self.load_results()
```

#### 分析方法

##### load_results()
```python
def load_results(self):
    """从目录加载所有实验结果"""
```

从 `results/` 目录加载：
- CSV文件
- pickle文件
- JSON配置

##### compute_statistics()
```python
def compute_statistics(self) -> Dict:
    """计算基本统计"""
```

返回：
```python
{
    'total_runs': 8192,
    'detection_rate_mean': 0.073,
    'detection_rate_std': 0.125,
    'detection_by_bit': pd.Series(...),
    'detection_by_location': pd.Series(...),
    'detection_by_layer': pd.Series(...)
}
```

##### analyze_by_bit_position()
```python
def analyze_by_bit_position(self) -> pd.DataFrame:
    """按比特位分析检测率"""
```

返回DataFrame：
```
bit | mean_detection | std | count | min | max
----|----------------|-----|-------|-----|-----
 0  | 0.000          | 0.0 | 256   | 0.0 | 0.0
 1  | 0.000          | 0.0 | 256   | 0.0 | 0.0
...
23  | 0.150          | 0.2 | 256   | 0.0 | 0.8
...
30  | 0.734          | 0.3 | 256   | 0.2 | 1.0
31  | 0.520          | 0.4 | 256   | 0.0 | 1.0
```

##### generate_report()
```python
def generate_report(self, output_path: str = "analysis_report.txt"):
    """生成文本分析报告"""
```

**报告内容**：
```
============================================================
Experiment Analysis Report
============================================================
Generated: 2025-12-19 10:30:00

Overall Statistics:
  Total runs: 8192
  Mean detection rate: 7.3%
  Std detection rate: 12.5%

Detection by Bit Position:
  Bit  0-22: 0.0% - 0.5%
  Bit 23-29: 10.0% - 30.0%
  Bit 30-31: 70.0% - 75.0%

Detection by Location:
  scores:  5.2%
  weights: 8.4%
  q:       7.1%
  k:       6.9%

Detection by Layer:
  Layer 0: 7.5%
  Layer 3: 7.2%
  Layer 6: 7.1%
  Layer 9: 7.4%

Top 10 Configurations:
  1. seed=42, layer=6, bit=30, location=weights: 75.2%
  2. seed=123, layer=9, bit=31, location=out: 74.8%
  ...

Bound Type Analysis:
  s@w:  6.2%
  q@o:  6.4%
  comb: 7.3%

Force KV Equal Analysis:
  False: 4.5%
  True:  7.3%
```

#### 使用示例

```python
from test.analyzer import ExperimentAnalyzer

# 加载结果
analyzer = ExperimentAnalyzer("./results/gpt2_sweep_20251219_103000")

# 计算统计
stats = analyzer.compute_statistics()
print(f"Overall detection rate: {stats['detection_rate_mean']:.2%}")

# 按比特位分析
bit_analysis = analyzer.analyze_by_bit_position()
print(bit_analysis)

# 生成报告
analyzer.generate_report("./analysis_report.txt")
```

---

### 3. visualizer.py - 结果可视化

**功能**：生成实验结果的可视化图表。

#### 核心类

##### ExperimentVisualizer
```python
class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self.setup_style()
```

#### 可视化方法

##### plot_detection_by_bit()
```python
def plot_detection_by_bit(self, save_path: str = None):
    """绘制按比特位的检测率（条形图）"""
```

**输出**：显示比特位0-31的检测率柱状图，高位比特明显更高。

##### plot_detection_heatmap()
```python
def plot_detection_heatmap(
    self,
    x_dim: str = 'injection_bit',
    y_dim: str = 'injection_location',
    save_path: str = None
):
    """绘制检测率热图"""
```

**输出示例**：
```
        Bit 0  Bit 7  Bit 15  Bit 23  Bit 30  Bit 31
scores    0%     0%      1%     15%     73%     50%
weights   0%     0%      2%     20%     75%     55%
q         0%     0%      1%     18%     70%     48%
k         0%     0%      1%     17%     69%     47%
```

##### plot_bound_type_comparison()
```python
def plot_bound_type_comparison(self, save_path: str = None):
    """对比不同bound_type的检测率"""
```

**输出**：条形图对比 s@w、q@o、comb 三种方法的检测率。

##### plot_force_kv_comparison()
```python
def plot_force_kv_comparison(self, save_path: str = None):
    """对比force_kv_equal的影响"""
```

**输出**：对比图显示 force_kv_equal=True/False 的检测率差异。

##### create_summary_dashboard()
```python
def create_summary_dashboard(self, save_path: str = None):
    """创建综合仪表盘"""
```

**仪表盘布局**：
```
┌─────────────────────────────────────────────┐
│  总体统计  │  比特位检测率  │  Top配置    │
├─────────────────────────────────────────────┤
│  位置热图            │  层分析图          │
├─────────────────────────────────────────────┤
│  Bound类型对比       │  KV对比图          │
└─────────────────────────────────────────────┘
```

#### 使用示例

```python
from test.analyzer import ExperimentAnalyzer
from test.visualizer import ExperimentVisualizer

# 加载结果
analyzer = ExperimentAnalyzer("./results/gpt2_sweep")
df = analyzer.results_df

# 创建可视化器
viz = ExperimentVisualizer(df)

# 生成图表
viz.plot_detection_by_bit("detection_by_bit.png")
viz.plot_detection_heatmap("heatmap.png")
viz.plot_bound_type_comparison("bound_type_comparison.png")
viz.plot_force_kv_comparison("force_kv_comparison.png")
viz.create_summary_dashboard("dashboard.png")
```

---

## 完整工作流

### 1. 运行实验（复现论文结果）

```bash
cd test
python run_experiment.py sweep
```

**预期运行时间**：
- 8192个配置
- 每个配置约5-10秒
- 总计：约11-22小时（取决于硬件）

**可选：快速验证**
```python
# 修改 run_experiment.py 中的扫描参数
sweep_params={
    'seed': [42],                    # 只用1个seed
    'injection_bit': [23, 30, 31],   # 只测试3个高位比特
    'injection_location': ['scores'], # 只测试1个位置
    'injection_idx': [(0, 0, 10, 20)] # 只测试1个空间位置
}
# 总配置: 1 × 4 × 3 × 1 × 1 = 12
# 运行时间: 约1-2分钟
```

### 2. 分析结果

```python
from test.analyzer import ExperimentAnalyzer

analyzer = ExperimentAnalyzer("./results/gpt2_sweep_20251219_103000")

# 生成报告
analyzer.generate_report("analysis_report.txt")

# 查看统计
stats = analyzer.compute_statistics()
print(f"Detection rate: {stats['detection_rate_mean']:.2%}")

# 按比特位分析
bit_stats = analyzer.analyze_by_bit_position()
print("\n高位比特检测率:")
print(bit_stats.loc[23:31])
```

### 3. 可视化

```python
from test.visualizer import ExperimentVisualizer

viz = ExperimentVisualizer(analyzer.results_df)

# 生成所有图表
viz.plot_detection_by_bit("figures/bit_detection.png")
viz.plot_detection_heatmap("figures/heatmap.png")
viz.plot_bound_type_comparison("figures/bound_type.png")
viz.plot_force_kv_comparison("figures/force_kv.png")
viz.create_summary_dashboard("figures/dashboard.png")
```

---

## 参数配置快速参考

### 在 run_experiment.py 中修改

#### 1. bound_type（边界计算方法）
```python
# 第258行附近
base_config = ExperimentConfig(
    bound_type="comb",  # ⭐ 修改此处
)
```

选项：
- `"s@w"`: scores × weights 路径（通用，K≠V可用）
- `"q@o"`: query × output 路径（需要K=V）
- `"comb"`: 组合两种路径（推荐，最高检测率）

#### 2. force_kv_equal（强制KV相等）
```python
# 第368行附近
model = monkey_patch_model(
    model, 'gpt2',
    force_kv_equal=True  # ⭐ 修改此处
)
```

选项：
- `False`: 不强制（使用原始权重）
- `True`: 强制K=V（推荐，检测率更高）

#### 3. 扫描参数范围
```python
# 第285-312行
sweep_params={
    'seed': [0, 42, 123, 3407],              # ⭐ 修改种子数
    'injection_bit': list(range(32)),         # ⭐ 修改比特范围
    'injection_location': [...],              # ⭐ 修改位置列表
}
```

---

## 输出文件结构

```
results/
├── gpt2_sweep_20251219_103000/
│   ├── config.json                  # 基础配置
│   ├── logs/
│   │   └── experiment.log           # 运行日志
│   ├── violations/
│   │   ├── violation_log.txt        # 违反记录（文本）
│   │   └── violation_summary.json   # 违反统计（JSON）
│   └── figures/                     # 可视化图表（如果生成）
│       ├── detection_by_bit.png
│       ├── heatmap.png
│       └── dashboard.png
│
└── analysis_report.txt              # 分析报告
```

---

## 最佳实践

### 1. 快速验证
```bash
# 修改为小规模扫描
python run_experiment.py simple
```

### 2. 完整复现
```bash
python run_experiment.py sweep
# 等待完成（~12小时）
```

### 3. 中断和恢复
- 实验会自动保存每个配置的结果
- 如果中断，可以记录最后完成的配置索引
- 修改代码跳过已完成的配置

### 4. 并行运行
- 可以修改代码支持多进程
- 或在不同机器上运行不同的seed组

---

## 故障排除

### 问题1: CUDA out of memory
```python
# 减小batch_size
config = ExperimentConfig(batch_size=2)  # 默认4

# 或减小seq_length
config = ExperimentConfig(seq_length=64)  # 默认128
```

### 问题2: 运行太慢
```python
# 减少扫描范围
sweep_params={
    'injection_bit': [23, 24, 30, 31],  # 只测试关键比特
    'seed': [42],                        # 只用一个seed
}
```

### 问题3: 找不到结果文件
```python
# 检查save_dir
config = ExperimentConfig(
    save_dir="./results"  # 确保路径正确
)

# 或使用绝对路径
analyzer = ExperimentAnalyzer("/absolute/path/to/results")
```

---

## 扩展建议

### 添加新的实验类型
1. 在 `run_experiment.py` 中添加新函数
2. 注册到 `__main__` 的选项中
3. 遵循现有的命名和结构

### 添加新的分析维度
1. 在 `analyzer.py` 中添加分析方法
2. 在 `visualizer.py` 中添加可视化
3. 更新 `generate_report()` 包含新指标