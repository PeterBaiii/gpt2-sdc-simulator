# utils/ 目录架构说明

## 概述

`utils/` 目录包含辅助工具函数，提供调试输出、NaN检测、边界诊断等功能。这些工具在开发、调试和诊断实验问题时非常有用。

## 文件列表

### 1. debug.py - 调试输出工具

**功能**：提供增强的调试输出功能，支持彩色输出、日志记录、时间戳等。

#### 核心类

##### DebugConfig
```python
class DebugConfig:
    """全局配置类"""
    enabled = True              # 是否启用debug
    use_color = True            # 是否使用彩色输出
    log_to_file = False         # 是否写入文件
    log_file_path = "debug.log" # 日志文件路径
    show_timestamp = True       # 是否显示时间戳
```

##### Colors
```python
class Colors:
    """ANSI颜色代码"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
```

#### 核心函数

##### debug()
```python
def debug(*args, **kwargs):
    """
    增强的调试输出函数
    
    参数:
        *args: 要输出的内容
        color: 颜色 ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan')
        level: 日志级别 ('INFO', 'WARNING', 'ERROR', 'DEBUG')
    """
```

**使用示例**：
```python
from utils.debug import debug

# 基本使用
debug("检查边界计算", color="cyan")

# 输出变量
bounds = compute_bounds(...)
debug("Bounds computed:", "epsilon mean =", bounds.epsilon.mean().item())

# 不同级别
debug("实验开始", level="INFO", color="blue")
debug("检测到违反", level="WARNING", color="yellow")
debug("致命错误", level="ERROR", color="red")
```

**输出格式**：
```
[2025-12-19 10:30:45] [DEBUG] 检查边界计算
[2025-12-19 10:30:46] [INFO] 实验开始
[2025-12-19 10:30:47] [WARNING] 检测到违反
```

##### 配置函数

```python
def enable_debug(enabled: bool = True):
    """启用/禁用debug输出"""

def enable_log_file(file_path: str = "debug.log"):
    """启用文件日志"""

def set_color_output(enabled: bool = True):
    """设置是否使用彩色输出"""

def set_timestamp(enabled: bool = True):
    """设置是否显示时间戳"""
```

---

### 2. check_nan.py - NaN检测工具

**功能**：检测张量中的NaN值并提供详细的位置信息。

#### 核心函数

##### check_nan()
```python
def check_nan(
    tensor: Union[torch.Tensor, np.ndarray],
    name: str = "tensor",
    max_display: int = 10,
    return_positions: bool = False,
    verbose: bool = True
) -> Tuple[bool, List[Tuple]]:
    """检测张量中的NaN值"""
```

**使用示例**：

```python
from utils.check_nan import check_nan

# 基本检测
scores = compute_scores(...)
check_nan(scores, name="attention_scores")

# 获取位置列表
has_nan, positions = check_nan(
    weights,
    name="attention_weights",
    return_positions=True
)

if has_nan:
    print(f"Found NaN at {len(positions)} positions")
    # 进行修复...
```

**输出示例**：
```
⚠️  [NaN检测] 张量 'attention_scores' 中发现 3 个NaN值！
张量形状: torch.Size([4, 12, 128, 128])
张量dtype: torch.float32
NaN位置 (显示前10个):
  #1: (b=0, c=3, h=45, w=67)
  #2: (b=1, c=7, h=23, w=89)
  #3: (b=2, c=11, h=78, w=12)
```

##### check_nan_summary()
```python
def check_nan_summary(tensor, name: str = "tensor") -> dict:
    """返回NaN检测的摘要信息"""
```

返回：
```python
{
    'has_nan': True,
    'nan_count': 15,
    'total_elements': 196608,
    'nan_ratio': 0.000076,
    'shape': (4, 12, 128, 128),
    'dtype': 'torch.float32'
}
```

##### assert_no_nan()
```python
def assert_no_nan(tensor, name: str = "tensor"):
    """断言张量中不包含NaN，否则抛出异常"""
```

**使用场景**：严格检查关键计算步骤

```python
# 边界计算前验证
assert_no_nan(scores, "scores")
assert_no_nan(weights, "weights")

bounds = compute_attention_bounds(scores, weights, d)

# 边界计算后验证
assert_no_nan(bounds.epsilon, "epsilon")
assert_no_nan(bounds.lower1, "lower1")
assert_no_nan(bounds.upper, "upper")
```

---

### 3. bound_fixing.py - 边界诊断工具

**功能**：分析边界不等式是否成立，诊断违反情况。

#### 核心函数

##### hist_tensor_diff()
```python
def hist_tensor_diff(
    bounds: Dict[str, torch.Tensor],
    eps: float = 1e-6,
    bins: Optional[List[float]] = None,
    max_neg_show: int = 10
):
    """
    分析边界不等式的差值分布
    
    检查三组不等式:
    1. lower1 ≤ middle
    2. middle ≤ epsilon
    3. epsilon ≤ upper
    """
```

**使用示例**：

```python
from utils.bound_fixing import hist_tensor_diff

# 计算边界
bounds_result = compute_attention_bounds(scores, p, d)
bounds_dict = bounds_result.to_dict()

# 分析边界
hist_tensor_diff(bounds_dict, eps=1e-6)
```

**输出示例**：
```
=== lower1 → middle ===
Total elements: 2048
Min diff: -0.001234, Max diff: 5.678901
eps tolerance: 1e-06

No NaN/Inf values detected.

Value range          Count    Percent
[0.0, 0.001)           150     7.32%
[0.001, 0.01)          450    21.97%
[0.01, 0.1)            800    39.06%
[0.1, 1.0)             600    29.30%
[1.0, ∞)                48     2.34%

⚠️  Found 12 negative diffs → violate inequality!
  Negative diff range: [-0.001234, -0.000001]
  Showing up to 10 violations:
    idx=(0, 0, 5, 8), lower1=2.345, middle=2.344, diff=-0.001

=== middle → epsilon ===
...

=== epsilon → upper ===
...
```

**典型用例**：

```python
# 诊断边界问题
results = run_experiment(config)

if results.detection_rate < 0.01:
    print("检测率异常低，诊断边界...")
    hist_tensor_diff(results.bounds_dict)
    
    # 可能的问题:
    # - 边界太松 (所有diff都很大)
    # - 输入数据异常 (大量NaN)
    # - 计算精度问题 (很多小负值)
```

---

### 4. return_top2.py - Top-2分析

**功能**：分析注意力分数的top-2模式，帮助理解margin。

```python
def analyze_top2_pattern(
    scores: torch.Tensor,
    p: torch.Tensor
):
    """
    分析每个查询位置的top-2注意力分数
    
    返回:
        {
            'j_star': 最大位置索引
            'second_max_j': 第二大位置索引
            'gamma': margin (max - second_max)
            'w_star': 最大权重
            'w_second': 第二大权重
        }
    """
```

**使用示例**：
```python
from utils.return_top2 import analyze_top2_pattern

top2_info = analyze_top2_pattern(scores, weights)

# 查看margin分布
print(f"Gamma min: {top2_info['gamma'].min()}")
print(f"Gamma mean: {top2_info['gamma'].mean()}")
print(f"Gamma max: {top2_info['gamma'].max()}")

# 找出margin最小的位置（难以检测）
small_margin_mask = top2_info['gamma'] < 0.1
problematic_positions = torch.nonzero(small_margin_mask)
print(f"Small margin positions: {len(problematic_positions)}")
```

---

## 工具组合使用

### 完整调试工作流

```python
from utils.debug import debug, enable_log_file
from utils.check_nan import check_nan, assert_no_nan
from utils.bound_fixing import hist_tensor_diff

# 1. 启用日志
enable_log_file("experiment_debug.log")

# 2. 运行实验
debug("开始运行实验...", color="cyan")

try:
    # 3. 检查输入
    debug("检查输入张量", color="blue")
    assert_no_nan(scores, "scores")
    assert_no_nan(p, "weights")
    
    # 4. 计算边界
    debug("计算边界...", color="blue")
    bounds = compute_attention_bounds(scores, p, d)
    
    # 5. 检查边界
    debug("检查边界有效性", color="blue")
    for key in ['lower1', 'middle', 'epsilon', 'upper']:
        check_nan(bounds.to_dict()[key], name=key)
    
    # 6. 分析边界
    debug("分析边界不等式", color="blue")
    hist_tensor_diff(bounds.to_dict())
    
    # 7. 检测违反
    violation = detect_violation(bounds, injected_epsilon)
    
    if violation['violated']:
        debug("检测到违反!", color="green", level="INFO")
    else:
        debug("未检测到违反", color="yellow", level="WARNING")
        
except Exception as e:
    debug("实验失败!", color="red", level="ERROR")
    debug(f"错误: {e}", color="red")
    raise

debug("实验完成", color="green")
```

### 批量实验诊断

```python
from utils.debug import debug, set_timestamp
from utils.check_nan import check_nan_summary
from utils.bound_fixing import hist_tensor_diff
from pathlib import Path
import pickle

# 禁用时间戳让输出更简洁
set_timestamp(False)

results_dir = Path("./results")
nan_stats = []

for exp_dir in results_dir.glob("exp_*"):
    debug(f"\n分析 {exp_dir.name}", color="cyan")
    
    # 加载结果
    with open(exp_dir / "bounds.pkl", "rb") as f:
        bounds = pickle.load(f)
    
    # NaN统计
    for key in ['epsilon', 'lower1', 'upper']:
        summary = check_nan_summary(bounds[key], key)
        if summary['has_nan']:
            nan_stats.append({
                'exp': exp_dir.name,
                'tensor': key,
                'nan_ratio': summary['nan_ratio']
            })
    
    # 边界分析
    hist_tensor_diff(bounds, max_neg_show=1)

# 汇总报告
print(f"\n{'='*60}")
print(f"共分析 {len(list(results_dir.glob('exp_*')))} 个实验")
print(f"发现 {len(nan_stats)} 个包含NaN的张量")
print(f"{'='*60}")
```

---

## 使用场景

### 1. 开发阶段

```python
from utils.debug import debug
from utils.check_nan import check_nan

def develop_new_bounds():
    """开发新的边界计算函数"""
    
    debug("测试新边界公式", color="cyan")
    
    # 生成测试数据
    scores = torch.randn(2, 4, 8, 8)
    weights = F.softmax(scores, dim=-1)
    
    # 检查输入
    check_nan(scores, "scores")
    check_nan(weights, "weights")
    
    # 计算新边界
    new_bounds = compute_new_bounds(scores, weights)
    
    # 验证结果
    debug("验证不等式", color="blue")
    check_nan(new_bounds['lower'], "lower")
    check_nan(new_bounds['upper'], "upper")
    
    # 诊断
    from utils.bound_fixing import hist_tensor_diff
    hist_tensor_diff(new_bounds)
```

### 2. 调试实验

```python
# 在 run_experiment.py 中
from utils.debug import debug, enable_log_file
from utils.check_nan import check_nan

# 启用详细日志
enable_log_file("debug.log")

def run_single_experiment():
    debug("="*60, color="cyan")
    debug("开始实验", color="cyan")
    
    # Baseline
    debug("运行baseline...", color="blue")
    baseline_output = model(inputs)
    check_nan(baseline_output, "baseline_output")
    
    # Injected
    debug("运行injected...", color="blue")
    injected_output = model_with_injection(inputs)
    check_nan(injected_output, "injected_output")
    
    # 边界
    debug("计算边界...", color="blue")
    bounds = compute_bounds(...)
    check_nan(bounds.epsilon, "epsilon")
    
    debug("="*60, color="green")
```

### 3. 诊断异常结果

```python
# 当检测率异常低时
if detection_rate < 0.01:
    from utils.bound_fixing import hist_tensor_diff
    from utils.debug import debug
    
    debug("检测率异常低，进行诊断", color="yellow", level="WARNING")
    
    # 检查边界
    hist_tensor_diff(bounds_dict)
    
    # 检查NaN
    for name, tensor in intermediate_tensors.items():
        summary = check_nan_summary(tensor)
        if summary['has_nan']:
            debug(f"{name} contains {summary['nan_count']} NaNs!",
                  color="red", level="ERROR")
```

---

## 最佳实践

### 1. 开发阶段
- 使用 `debug()` 追踪关键变量
- 用 `check_nan()` 验证每个计算步骤
- 启用文件日志保存调试信息

```python
from utils.debug import debug, enable_log_file
enable_log_file("dev_debug.log")
```

### 2. 测试阶段
- 使用 `assert_no_nan()` 做严格检查
- 用 `hist_tensor_diff()` 验证边界正确性
- 对比不同配置的诊断结果

```python
# 严格模式
assert_no_nan(scores, "scores")
assert_no_nan(weights, "weights")
```

### 3. 生产阶段
- 禁用详细debug输出（`enable_debug(False)`）
- 保留关键的NaN检查
- 使用摘要函数记录统计信息

```python
from utils.debug import enable_debug
enable_debug(False)  # 生产环境禁用

# 但保留关键检查
from utils.check_nan import check_nan_summary
summary = check_nan_summary(bounds.epsilon)
if summary['has_nan']:
    log_error(f"NaN detected: {summary}")
```

### 4. 性能考虑
- debug和check_nan可能有开销
- 在性能关键路径上谨慎使用
- 使用 `verbose=False` 减少输出
- 考虑采样检查而非全量检查

```python
# 采样检查（每10个batch检查一次）
if batch_idx % 10 == 0:
    check_nan(outputs, f"outputs_batch_{batch_idx}")
```

---

## 配置建议

### 调试配置

```python
# 开发/调试配置
from utils.debug import (
    enable_log_file, 
    set_timestamp, 
    set_color_output
)

enable_log_file("debug.log")
set_timestamp(True)
set_color_output(True)
```

### 生产配置

```python
# 生产配置
from utils.debug import (
    enable_debug,
    enable_log_file,
    set_timestamp
)

enable_debug(False)           # 禁用debug输出
enable_log_file("errors.log") # 只记录错误
set_timestamp(True)           # 保留时间戳
```

---

## 扩展建议

### 添加新的诊断工具

```python
# utils/tensor_stats.py
def compute_tensor_stats(tensor: torch.Tensor) -> dict:
    """计算张量统计信息"""
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item()
    }

# 使用
from utils.tensor_stats import compute_tensor_stats
stats = compute_tensor_stats(bounds.epsilon)
print(f"Epsilon stats: {stats}")
```

### 集成到实验框架

```python
# 在 ExperimentRunner 中集成
class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        
        # 根据配置启用调试
        if config.debug_mode:
            from utils.debug import enable_log_file
            enable_log_file(f"{config.save_dir}/debug.log")
    
    def run_sample(self, inputs, injection_config):
        # 自动NaN检查
        if self.config.debug_mode:
            from utils.check_nan import check_nan
            check_nan(inputs, "inputs", verbose=True)
        
        # 运行
        result = self._forward(inputs, injection_config)
        
        # 诊断
        if self.config.debug_mode and not result.violation_detected:
            from utils.bound_fixing import hist_tensor_diff
            hist_tensor_diff(result.bounds.to_dict())
        
        return result
```

---

## 工具对照表

| 工具 | 用途 | 使用场景 | 开销 |
|------|------|----------|------|
| `debug()` | 彩色日志输出 | 追踪执行流程 | 低 |
| `check_nan()` | NaN检测 | 验证计算正确性 | 中 |
| `assert_no_nan()` | 严格NaN检查 | 关键步骤验证 | 中 |
| `hist_tensor_diff()` | 边界诊断 | 分析不等式违反 | 中-高 |
| `analyze_top2_pattern()` | Top-2分析 | 理解margin分布 | 低-中 |

**建议**：
- 开发阶段：全部使用
- 测试阶段：使用debug + check_nan + hist_tensor_diff
- 生产阶段：只用check_nan_summary（采样）