# src/ 目录架构说明

## 概述

`src/` 目录包含项目的核心实现代码，负责模型适配、故障注入、边界计算、实验配置和执行等关键功能。

## 参数配置概览 ⭐

### 边界计算方法 (bound_type)

项目支持三种边界计算方法，在 `ExperimentConfig` 中通过 `bound_type` 参数设置：

#### 1. "s@w" - Scores × Weights路径

```python
config = ExperimentConfig(bound_type="s@w")
```

**计算方式**：
```python
εi = √d · Σj wij·aij
```
其中：
- `aij = qi·kj / √d`：attention logits
- `wij = softmax(aij)`：attention weights

**适用场景**：
- K≠V 的通用情况
- 不需要强制K=V权重相等
- 检测率：~4.5%（K≠V）、~6.2%（K=V）

**实现位置**：
- `src/bounds_computation.py` 中的 `compute_injected_epsilon_from_p()`
- `src/experiment_runner.py` 第465-468行

#### 2. "q@o" - Query × Output路径

```python
config = ExperimentConfig(bound_type="q@o")
```

**计算方式**：
```python
εi = qi · Attn(xi)
```
其中：
- `qi`：query向量
- `Attn(xi) = Σj wij·vj`：attention输出

**适用场景**：
- 需要 K=V 假设
- 需要设置 `force_kv_equal=True`
- 检测率：~6.4%（K=V）

**实现位置**：
- `src/bounds_computation.py` 中的 `compute_injected_epsilon()`
- `src/experiment_runner.py` 第471-475行

#### 3. "comb" - 组合路径 ⭐ 推荐

```python
config = ExperimentConfig(bound_type="comb")
```

**计算方式**：
- 同时计算 s@w 和 q@o 两种路径
- 任一路径违反边界即报告检测

**适用场景**：
- 最佳检测率配置
- 需要 K=V 假设
- 检测率：~7.3%（K=V）、24.7%（高位比特）

**实现位置**：
- `src/experiment_runner.py` 第465-477行
- 同时调用两个计算函数

### 强制KV相等 (force_kv_equal)

在调用 `monkey_patch_model()` 时设置：

```python
from src.model_adapter import monkey_patch_model

model = monkey_patch_model(
    model,
    'gpt2',
    injection_layers=[0, 3, 6, 9],
    force_kv_equal=True  # ⭐ 关键参数
)
```

**效果**：
- `False`：使用模型原始权重（K和V可能不同）
- `True`：强制令 K 权重等于 V 权重（通过 `force_kv_consistent()`）

**实现细节**：
```python
# model_adapter.py 第495-523行
@torch.no_grad()
def force_kv_consistent(model, mode: str = "K<-V"):
    """
    强制K、V权重相等
    
    mode: "K<-V" 表示令 K 等于 V
          "V<-K" 表示令 V 等于 K
    """
    for layer in model.transformer.h:
        W = layer.attn.c_attn.weight
        s = layer.attn.split_size
        
        # K权重区间 [s:2s] 复制为 V权重区间 [2s:3s]
        if mode.upper() == "K<-V":
            W[:, s:2*s].copy_(W[:, 2*s:3*s])
```

**推荐配置组合**：
```python
# 最佳检测率
force_kv_equal=True + bound_type="comb"

# 平衡速度
force_kv_equal=True + bound_type="s@w"
```

---

## 模块列表

### 1. model_adapter.py - 模型适配器

**功能**：为不同的Transformer模型提供统一的注意力层访问接口。

#### 核心类

##### AttentionHook
```python
class AttentionHook:
    """
    注意力层Hook
    
    职责：
    1. 执行错误注入（如果配置了）
    2. 返回中间张量的引用（不存储副本）
    """
```

**关键方法**：
- `register_tensor(name, tensor)`: 注册中间张量
- `get_tensors()`: 获取所有注册的张量
- `maybe_inject(name, tensor)`: 条件性执行比特翻转注入
- `reset()`: 重置Hook状态

##### BaseAttentionAdapter (抽象基类)
```python
class BaseAttentionAdapter(ABC):
    """注意力层适配器基类"""
    
    @abstractmethod
    def forward_with_injection(
        self, hidden_states, attention_mask=None,
        injection_config=None, return_intermediates=True
    ):
        """带注入的前向传播"""
```

##### GPT2AttentionAdapter
```python
class GPT2AttentionAdapter(BaseAttentionAdapter):
    """GPT-2注意力层适配器"""
```

**前向传播流程**：
```python
1. QKV投影:     qkv = self.attn.c_attn(hidden_states)
2. 分割:        q, k, v = split(qkv)
3. Reshape:     q/k/v -> (B, H, L, D)
4. 注入 q/k/v
5. 计算scores:  attn_weights = q @ k.T / √d
6. 注入 scores
7. 应用mask
8. Softmax:     weights = softmax(attn_weights)
9. 注入 weights
10. 输出:       output = weights @ v
11. 注入 out
12. Merge:      output -> (B, L, hidden)
13. 输出投影
```

#### 工具函数

##### monkey_patch_model()
```python
def monkey_patch_model(
    model, model_type: str,
    injection_layers: Optional[List[int]] = None,
    force_kv_equal: bool = False  # ⭐ 关键参数
):
    """对模型进行monkey patch"""
```

**功能**：
1. 为指定层创建adapter对象
2. 保存到 `layer.attn.adapter`
3. 记录 `layer.attn.layer_idx`
4. 初始化 `layer.attn._injection_config`
5. **如果 force_kv_equal=True，强制K=V权重相等**

**使用示例**：
```python
# 不强制K=V（使用原始权重）
model = monkey_patch_model(model, 'gpt2', force_kv_equal=False)

# 强制K=V（实验发现这样检测率更高）
model = monkey_patch_model(model, 'gpt2', force_kv_equal=True)
```

##### check_kv_consistency()
```python
def check_kv_consistency(model, model_type: str) -> Dict:
    """检查模型的K和V权重是否一致"""
```

返回每层的KV一致性信息。

##### force_kv_consistent()
```python
@torch.no_grad()
def force_kv_consistent(model, mode: str = "K<-V"):
    """强制所有层的K、V权重相等"""
```

**实现细节**：
```python
for layer in model.transformer.h:
    W = layer.attn.c_attn.weight  # (hidden, 3*hidden)
    s = layer.attn.split_size
    
    # Q: [0:s], K: [s:2s], V: [2s:3s]
    if mode == "K<-V":
        # 将K权重复制为V权重
        W[:, s:2*s].copy_(W[:, 2*s:3*s])
```

---

### 2. fault_injection.py - 故障注入

**功能**：定义注入配置和比特翻转操作。

#### 核心类

##### InjectionLocation (枚举)
```python
class InjectionLocation(Enum):
    Q = "q"
    K = "k"
    V = "v"
    SCORES = "scores"
    WEIGHTS = "weights"
    OUT = "out"
```

##### InjectionConfig
```python
@dataclass
class InjectionConfig:
    location: InjectionLocation
    idx: tuple                   # (b, h, i, j)
    bit: int                     # [0, 31]
    enabled: bool = True
```

#### 核心函数

##### bitflip_()
```python
def bitflip_(tensor: torch.Tensor, idx: tuple, bit: int):
    """原地翻转张量指定位置的比特"""
```

**IEEE 754 布局**：
```
bit 31: 符号位
bit 30-23: 指数位 (8 bits)
bit 22-0: 尾数位 (23 bits)
```

---

### 3. bounds_computation.py - 边界计算

**功能**：计算注意力机制的理论上下界，检测违反。

#### 核心函数

##### compute_attention_bounds()
```python
@torch.no_grad()
def compute_attention_bounds(
    scores: torch.Tensor,    # (B, H, T, T)
    p: torch.Tensor,         # (B, H, T, T)
    d: int,                  # head_dim
    handle_nan: bool = True,
    causal_mask: Optional[torch.Tensor] = None
) -> BoundsResult:
```

**计算步骤**：
1. 查找最大值和margin
2. 计算下界：`√d * γ / (1 + e^γ)`
3. 计算中间值（均值下界）
4. 计算上界（Lambert-W）
5. 返回 `BoundsResult` 对象

##### compute_injected_epsilon_from_p() - s@w路径 ⭐
```python
def compute_injected_epsilon_from_p(
    scores: torch.Tensor,      # (B, H, T, T)
    p: torch.Tensor,           # (B, H, T, T)
    d: int
) -> torch.Tensor:
    """使用 scores × weights 路径计算 εi"""
```

**计算**：
```python
epsilon = sqrt_d * (p * scores).sum(dim=-1)
```

**使用场景**：
- `bound_type="s@w"` 或 `bound_type="comb"`
- 通用于K≠V和K=V情况

##### compute_injected_epsilon() - q@o路径 ⭐
```python
def compute_injected_epsilon(
    scores: torch.Tensor,       # (B, H, T, T)
    attn_output: torch.Tensor,  # (B, H, T, D)
    q: torch.Tensor,            # (B, H, T, D)
    d: int
) -> torch.Tensor:
    """使用 query × output 路径计算 εi"""
```

**计算**：
```python
epsilon = (q * attn_output).sum(dim=-1)
```

**使用场景**：
- `bound_type="q@o"` 或 `bound_type="comb"`
- 需要K=V假设

##### detect_violation()
```python
def detect_violation(
    bounds: BoundsResult,
    epsilon1: Optional[torch.Tensor],  # s@w路径
    epsilon2: Optional[torch.Tensor],  # q@o路径
    tolerance: float = 0.0
) -> Dict[str, Any]:
    """检测边界违反"""
```

**返回**：
```python
{
    'violated': bool,
    'violation_ratio': float,
    'lower_violations': int,
    'upper_violations': int,
    'injection_violations': {
        'any_violated': bool,
        'num_lower_violations': int,
        'num_upper_violations': int
    }
}
```

---

### 4. experiment_config.py - 实验配置

**功能**：统一管理所有实验参数。

#### 核心类

##### ExperimentConfig
```python
@dataclass
class ExperimentConfig:
    """实验总配置"""
```

**关键参数**：

```python
# ===== 基础配置 =====
exp_name: str = "default_exp"
seed: int = 42
device: str = "auto"

# ===== 边界检测配置 ===== ⭐
bounds_check: bool = True
tolerance: float = 0.0
bound_type: str = "s@w"  # "s@w" / "q@o" / "comb"

# ===== 注入配置 =====
injection_enabled: bool = True
injection_location: str = "scores"
injection_layers: Optional[List[int]] = None
injection_idx: tuple = (0, 0, 0, 0)
injection_bit: int = 15
```

**bound_type 说明**：
```python
# 方法1: scores × weights (通用)
bound_type = "s@w"

# 方法2: query × output (需要K=V)
bound_type = "q@o"

# 方法3: 组合两种路径 (推荐)
bound_type = "comb"
```

##### ParameterSweepConfig
```python
@dataclass
class ParameterSweepConfig:
    """参数扫描配置"""
    
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]]
```

**示例**：
```python
sweep_config = ParameterSweepConfig(
    base_config=base,
    sweep_params={
        'seed': [0, 42, 123, 3407],
        'injection_bit': list(range(32)),
        'injection_location': ['scores', 'weights', 'q', 'k'],
        'injection_layers': [[0], [3], [6], [9]]
    }
)
```

---

### 5. experiment_runner.py - 实验运行器

**功能**：执行完整实验流程，集成所有组件。

#### 核心类

##### IntermediateTensorCollector
```python
class IntermediateTensorCollector:
    """中间张量收集器"""
```

职责：
1. 从adapter返回的intermediates中收集张量
2. 存储所有层的张量副本（clone）
3. 提供统一的访问接口

##### CompleteExperimentRunner
```python
class CompleteExperimentRunner:
    """完整实验运行器"""
```

**核心方法**：

###### run_single()
```python
def run_single(self, run_id: int, model, dataloader) -> ExperimentResult:
    """运行单次实验"""
```

**关键流程**：

1. **Baseline运行**（无注入）
2. **Injected运行**（带注入）
3. **边界计算和检测** ⭐：

```python
# 第465-477行：处理不同的bound_type
if self.config.bound_type == "s@w" or self.config.bound_type == "comb":
    # 计算 s@w 路径
    weights_i = injected_layer['weights']
    injected_eps1 = compute_injected_epsilon_from_p(
        scores_i, weights_i, d
    )

if self.config.bound_type == "q@o" or self.config.bound_type == "comb":
    # 计算 q@o 路径
    attn_out_i = injected_layer['out']
    q_b = baseline_layer['q']
    injected_eps2 = compute_injected_epsilon(
        scores_i, attn_out_i, q_b, d
    )

# 检测违反
detection = detect_violation(
    bounds, 
    injected_eps1,  # s@w路径的epsilon
    injected_eps2,  # q@o路径的epsilon
    self.config.tolerance
)
```

4. **记录结果**

###### run_all()
```python
def run_all(self, model, dataloader) -> List[ExperimentResult]:
    """运行所有重复实验"""
```

---

### 6. experiment_logger.py - 实验日志

**功能**：统一的实验日志和结果记录。

#### 核心类

##### ExperimentResult
```python
@dataclass
class ExperimentResult:
    """单次实验结果"""
    
    # 基础信息
    exp_id: str
    run_id: int
    
    # 模型输出
    baseline_loss: Optional[float]
    injected_loss: Optional[float]
    loss_diff: Optional[float]
    
    # 边界检测结果
    violation_detected: Optional[bool]
    num_violations: Optional[int]
    
    # 注入信息
    injection_info: Optional[Dict]
```

##### ResultsLogger
```python
class ResultsLogger:
    """结果记录器"""
```

**方法**：
- `log(message)`: 记录日志
- `add_result(result)`: 添加结果
- `save()`: 保存所有结果

##### ViolationLogger
```python
class ViolationLogger:
    """违反记录器"""
```

专门记录检测到违反的配置，方便后续分析。

---

### 7. performance_monitor.py - 性能监控

**功能**：监控实验的运行时间和内存使用。

#### 核心类

##### PerformanceMonitor
```python
class PerformanceMonitor:
    """性能监控器"""
```

**方法**：
- `start_timer(name)`: 计时上下文管理器
- `get_memory_usage()`: 获取当前内存使用
- `get_metrics()`: 获取所有性能指标
- `print_summary()`: 打印性能摘要

---

## 参数配置最佳实践 ⭐

### 配置组合推荐

#### 1. 最佳检测率配置
```python
# 在 run_experiment.py 中设置
config = ExperimentConfig(
    bound_type="comb",        # 组合两种路径
    tolerance=0.0
)

model = monkey_patch_model(
    model, 'gpt2',
    force_kv_equal=True       # 强制K=V
)
```

**预期检测率**：
- 全体比特：~7.3%
- 高位比特（23-31）：~24.7%

#### 2. 平衡速度配置
```python
config = ExperimentConfig(
    bound_type="s@w",         # 单路径，更快
    tolerance=0.0
)

model = monkey_patch_model(
    model, 'gpt2',
    force_kv_equal=True
)
```

**预期检测率**：
- 全体比特：~6.2%
- 高位比特（23-31）：~21.0%

#### 3. 通用配置（不修改模型）
```python
config = ExperimentConfig(
    bound_type="s@w",         # K≠V通用方法
    tolerance=0.0
)

model = monkey_patch_model(
    model, 'gpt2',
    force_kv_equal=False      # 不修改权重
)
```

**预期检测率**：
- 全体比特：~4.5%
- 高位比特（23-31）：~16.1%

### 配置文件示例

**test/run_experiment.py 中的配置**：

```python
def run_parameter_sweep_experiment():
    # 基础配置
    base_config = ExperimentConfig(
        exp_name="gpt2_sweep",
        model_name="gpt2",
        batch_size=4,
        seq_length=128,
        num_samples=100,
        
        # ⭐ 关键配置
        bound_type="comb",        # 使用组合路径
        bounds_check=True,
        tolerance=0.0,
        
        # 注入配置
        injection_enabled=True,
        injection_location="scores",
        injection_layers=[0],
        injection_idx=(0, 0, 10, 20),
        injection_bit=23,
        
        seed=42
    )
    
    # 扫描参数
    sweep_config = ParameterSweepConfig(
        base_config=base_config,
        sweep_params={
            'seed': [0, 42, 123, 3407],
            'injection_layers': [[0], [3], [6], [9]],
            'injection_bit': list(range(32)),
            'injection_location': ['scores', 'weights', 'q', 'k'],
            'injection_idx': [
                (0, 0, 10, 20), (0, 3, 53, 43),
                (1, 6, 32, 1), (1, 9, 31, 62)
            ]
        }
    )
    
    # 加载模型
    model, tokenizer, device = load_gpt2_model('gpt2')
    
    # Patch模型
    for config in sweep_config.generate_configs():
        model = monkey_patch_model(
            model, 'gpt2',
            injection_layers=config.injection_layers,
            force_kv_equal=True  # ⭐ 强制K=V获得更好检测率
        )
        
        # 运行实验
        runner = CompleteExperimentRunner(config)
        results = runner.run_all(model, dataloader)
```

---

## 模块关系图

```
experiment_config.py (bound_type, tolerance)
        ↓
experiment_runner.py
        ↓
    根据 bound_type 选择计算路径
        ↓
┌───────┴───────┐
↓               ↓
s@w路径        q@o路径
↓               ↓
compute_       compute_
injected_      injected_
epsilon_       epsilon()
from_p()
        ↓
    detect_violation()
        ↓
    ExperimentResult
```

---

## 使用示例

### 修改bound_type

编辑 `test/run_experiment.py`：

```python
# 方法1: s@w
config = ExperimentConfig(bound_type="s@w")

# 方法2: q@o
config = ExperimentConfig(bound_type="q@o")

# 方法3: comb (推荐)
config = ExperimentConfig(bound_type="comb")
```

### 修改force_kv_equal

编辑 `test/run_experiment.py` 中的 `monkey_patch_model` 调用：

```python
# 不强制K=V
model = monkey_patch_model(model, 'gpt2', force_kv_equal=False)

# 强制K=V (推荐)
model = monkey_patch_model(model, 'gpt2', force_kv_equal=True)
```

### 检查当前模型的KV状态

```python
from src.model_adapter import check_kv_consistency

kv_check = check_kv_consistency(model, 'gpt2')
print(f"K=V globally: {kv_check['kv_equal']}")
for layer_info in kv_check['layer_info']:
    print(f"Layer {layer_info['layer_idx']}: {layer_info['kv_equal']}")
```

---

## 设计原则

1. **模块化**：每个文件专注于单一职责
2. **可配置**：所有参数通过配置对象管理
3. **可扩展**：通过适配器模式支持新模型
4. **轻量级**：最小化内存复制和计算开销
5. **可追溯**：完整记录配置和结果

## 扩展指南

### 添加新的bound_type

1. 在 `bounds_computation.py` 中添加新的计算函数
2. 在 `experiment_runner.py` 中添加新的分支
3. 在 `experiment_config.py` 中更新注释

### 添加新模型支持

1. 继承 `BaseAttentionAdapter`
2. 实现 `forward_with_injection()`
3. 在 `monkey_patch_model()` 中添加分支