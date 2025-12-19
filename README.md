# Silent Data Corruption Detection for Attention Mechanisms

基于蜕变关系边界的GPU静默数据损坏（SDC）检测框架，专门针对Transformer模型中的注意力机制。

## 项目简介

本项目实现了技术报告 *"Silent Data Corruption Detection Based on Metamorphic Attention Bounds"* 中描述的SDC检测方法。通过为自注意力层导出解析边界作为蜕变关系，在注入故障时检测GPU计算中的静默数据损坏。

### 核心特性

- **注意力边界计算**：为标准点积自注意力导出上下界
- **故障注入**：单比特翻转注入到注意力层中间张量
- **蜕变检查**：验证输出是否违反理论边界
- **多种边界计算方式**：支持 s@w、q@o、comb 三种计算方法
- **多模型支持**：支持GPT-2等Transformer模型
- **完整实验框架**：参数扫描、结果分析和可视化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 复现实验结果

**运行参数扫描实验（推荐）**：

```bash
cd test
python run_experiment.py sweep
```

或交互式选择：

```bash
cd test
python run_experiment.py
# 选择 2 (sweep) 进行参数扫描
```

这将自动运行完整的参数扫描实验，生成与论文一致的结果。

### 其他实验模式

```bash
# 运行简单单次实验
python run_experiment.py simple

# 运行多层注入实验
python run_experiment.py multilayer

# 运行所有实验
python run_experiment.py all
```

## 参数配置

### 核心配置说明

参数配置需要在 `test/run_experiment.py` 中的相应函数内修改。

#### 1. 边界计算方法 (bound_type)

在 `ExperimentConfig` 中设置：

```python
config = ExperimentConfig(
    # ...
    bound_type="comb",  # 边界计算方法
    # ...
)
```

**可选值**：
- `"s@w"`: 使用 scores × weights 路径计算（K≠V通用）
- `"q@o"`: 使用 Q × output 路径计算（需要K=V）
- `"comb"`: 组合两种路径，取并集（最佳检测率）

**说明**：
- `s@w` 路径计算: `εi = √d · Σj wij·aij`（使用logits和weights）
- `q@o` 路径计算: `εi = qi·Attn(xi)`（使用query和output）
- `comb` 模式会同时检查两种路径，任一违反即报告检测

#### 2. 强制KV相等 (force_kv_equal)

在 `monkey_patch_model` 调用时设置：

```python
model = monkey_patch_model(
    model,
    'gpt2',
    injection_layers=config.injection_layers,
    force_kv_equal=True  # 是否强制K=V
)
```

**说明**：
- `False`: 使用模型原始的K、V权重（可能不相等）
- `True`: 强制令K权重等于V权重（通过权重复制）

**实验发现**：
- K=V 假设下边界更紧，检测率更高
- 论文中最佳结果使用 `force_kv_equal=True` + `bound_type="comb"`

#### 3. 其他重要参数

```python
config = ExperimentConfig(
    # 基础配置
    exp_name="gpt2_sweep",           # 实验名称
    seed=42,                          # 随机种子
    
    # 模型配置
    model_name="gpt2",                # 模型名称
    
    # 数据配置
    dataset_name="wikitext",          # 数据集
    batch_size=4,                     # 批次大小
    seq_length=128,                   # 序列长度
    num_samples=100,                  # 样本数
    
    # 注入配置
    injection_enabled=True,           # 启用注入
    injection_location="scores",      # 注入位置
    injection_layers=[0],             # 注入层
    injection_idx=(0, 0, 10, 20),    # 张量索引
    injection_bit=23,                 # 比特位
    
    # 边界检测配置
    bounds_check=True,                # 启用边界检查
    bound_type="comb",                # 边界计算方法
    tolerance=0.0,                    # 违反容差
    
    # 实验控制
    num_runs=1,                       # 重复次数
    save_results=True,                # 保存结果
    save_dir="./results"              # 保存目录
)
```

### 参数扫描配置示例

在 `run_parameter_sweep_experiment()` 中配置扫描参数：

```python
from src.experiment_config import ParameterSweepConfig

sweep_config = ParameterSweepConfig(
    base_config=base_config,
    sweep_params={
        # 1. 随机种子
        'seed': [0, 42, 123, 3407],
        
        # 2. 注入层（选择不同的attention层）
        'injection_layers': [[0], [3], [6], [9]],
        
        # 3. 比特位（0-31）
        'injection_bit': list(range(32)),
        
        # 4. 注入位置（张量类型）
        'injection_location': ['scores', 'weights', 'q', 'k'],
        
        # 5. 空间位置
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

## 主要实验结果

根据GPT-2实验（参见PDF报告第16-20页）：

### 比特位检测率
- **高位比特** (23-31): 检测率显著提高
  - 比特30: ~73-75% 检测率
  - 比特23-29: 10-30% 检测率
- **低位比特** (0-22): 几乎无检测 (~0%)

### K=V vs K≠V 对比
- **K=V (force_kv_equal=True)**：
  - `s@w` 形式: 6.2% 总体检测率
  - `q@o` 形式: 6.4% 总体检测率
  - `comb` 组合: 7.3% 总体检测率
- **K≠V (force_kv_equal=False)**：
  - `s@w` 形式: 4.5% 总体检测率

### 指数/符号位检测（比特23-31）
- K≠V, s@w: 16.1% (371/2304)
- K=V, s@w: 21.0% (484/2304)
- K=V, q@o: 22.2% (512/2304)
- **K=V, comb: 24.7% (568/2304)** ← 最佳配置

### 运行时开销
- 注入开销: ~7% (几乎可忽略)
- 边界检查开销:
  - s@w 形式: ~13%
  - q@o 形式: ~20%
  - comb 形式: ~20%
- 内存开销: ~5%

## 目录结构

```
.
├── src/                          # 核心源代码
│   ├── bounds_computation.py     # 边界计算
│   ├── model_adapter.py          # 模型适配器
│   ├── fault_injection.py        # 故障注入
│   ├── experiment_config.py      # 实验配置
│   ├── experiment_runner.py      # 实验运行器
│   ├── experiment_logger.py      # 实验日志
│   └── performance_monitor.py    # 性能监控
│
├── test/                         # 测试和实验
│   ├── run_experiment.py         # 主实验脚本 ⭐
│   ├── analyzer.py               # 结果分析
│   └── visualizer.py             # 结果可视化
│
├── utils/                        # 工具函数
│   ├── bound_fixing.py           # 边界诊断
│   ├── check_nan.py              # NaN检查
│   ├── debug.py                  # 调试输出
│   └── return_top2.py            # Top-2分析
│
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

详细架构说明请参见：
- [src/ 架构文档](./docs/SRC_ARCH.md)
- [test/ 架构文档](./docs/TEST_ARCH.md)
- [utils/ 架构文档](./docs/UTILS_ARCH.md)

## 技术细节

### 注意力边界理论

对于自注意力层，定义：
- `aij = (qi · kj) / √d`：缩放点积分数
- `wij = softmax(aij)`：注意力权重
- `Attn(xi) = Σj wij · vj`：注意力输出

在K=V假设下，我们导出：
```
√d · γi / (1 + e^γi) ≤ εi ≤ min(qi·kj* - 1/n·Σj(qi·kj), √d·τ(γi, n))
```

其中：
- `γi`：softmax margin
- `εi = qi·kj* - qi·Attn(xi)`：偏差
- `τ(γi, n)`：Lambert-W上界函数

### 蜕变关系

我们使用两种等价的计算路径作为蜕变关系：
1. **输出路径** (`q@o`): `qi·Attn(xi) = Σj wij·(qi·kj)`
2. **逻辑路径** (`s@w`): `qi·Attn(xi) = √d·Σj Wij·Aij`

任何路径违反边界即标记为潜在SDC。

## 使用示例

### 修改实验配置

编辑 `test/run_experiment.py` 中的配置：

```python
def run_parameter_sweep_experiment():
    # 基础配置
    base_config = ExperimentConfig(
        exp_name="gpt2_custom_sweep",
        model_name="gpt2",
        batch_size=4,
        seq_length=128,
        num_samples=100,
        
        # 修改边界计算方法
        bound_type="comb",  # "s@w" / "q@o" / "comb"
        
        # 其他配置...
    )
    
    # 扫描参数
    sweep_config = ParameterSweepConfig(
        base_config=base_config,
        sweep_params={
            'injection_bit': [23, 24, 30, 31],  # 只测试高位比特
            'injection_location': ['scores', 'weights'],
            # ... 其他参数
        }
    )
    
    # 加载模型
    model, tokenizer, device = load_gpt2_model('gpt2')
    
    # Patch模型（设置force_kv_equal）
    for config in sweep_config.generate_configs():
        model = monkey_patch_model(
            model,
            'gpt2',
            injection_layers=config.injection_layers,
            force_kv_equal=True  # 修改此处控制K=V
        )
        # 运行实验...
```

### 分析结果

```python
from test.analyzer import ExperimentAnalyzer
from test.visualizer import ExperimentVisualizer

# 加载结果
analyzer = ExperimentAnalyzer("./results/gpt2_sweep")

# 生成统计报告
analyzer.generate_report("analysis_report.txt")

# 可视化
viz = ExperimentVisualizer(analyzer.results_df)
viz.plot_detection_by_bit("detection_by_bit.png")
viz.plot_detection_heatmap("heatmap.png")
viz.create_summary_dashboard("dashboard.png")
```

## 实验建议

### 快速验证（~2分钟）
```python
config = ExperimentConfig(
    batch_size=2,
    num_samples=10,
    injection_bit=30,  # 测试高位比特
    bound_type="comb"
)
```

### 完整复现（~20分钟）
```bash
cd test
python run_experiment.py sweep
```

参数扫描包括：
- 4个随机种子
- 4个注意力层
- 32个比特位
- 4个注入位置
- 4个空间位置
- **总计：8192个配置**

### 推荐配置

**最佳检测率**：
```python
force_kv_equal=True
bound_type="comb"
injection_bit in [23-31]  # 高位比特
```

**平衡速度和覆盖**：
```python
force_kv_equal=True
bound_type="s@w"
injection_bit in [23, 30]
num_samples=50
```

## 性能优化建议

1. **GPU内存**：使用`batch_size=4-8`以平衡速度和内存
2. **序列长度**：GPT-2在`seq_length=128`时运行良好
3. **采样数**：快速测试用10-50样本，完整实验用100+
4. **并行化**：多个seed可并行运行（修改代码支持多进程）

## 局限性

1. **覆盖率有限**：总体检测率~7-25%（取决于配置）
2. **比特选择性**：主要检测高位比特翻转（指数/符号位）
3. **单层注入**：当前实现每次只在一层注入
4. **假设依赖**：K=V假设提高检测但限制通用性

## 扩展方向

1. **更紧的边界**：改进理论推导获得更tight的上下界
2. **梯度边界**：支持训练时的反向传播检测
3. **多层同时注入**：模拟更复杂的故障场景
4. **其他架构**：支持BERT、LLaMA等模型
5. **混合精度**：支持FP16/BF16的边界计算

## 引用

如果使用本代码，请引用：

```bibtex
@article{bai2025sdc,
  title={Silent Data Corruption Detection Based on Metamorphic Attention Bounds},
  author={Bai, Xinyu},
  year={2025},
  institution={University of Illinois Urbana-Champaign}
}
```

## 相关工作

- [Dixit et al., 2021] - Silent Data Corruptions at Scale
- [Ma et al., 2025] - Understanding Silent Data Corruption in LLM Training
- [Hari et al., 2020] - Estimating Silent Data Corruption Rates Using a Two-Level Model

## 许可证

MIT License

## 联系方式

- 作者：Xinyu Bai
- 邮箱：xbai@illinois.edu
- 机构：University of Illinois Urbana-Champaign