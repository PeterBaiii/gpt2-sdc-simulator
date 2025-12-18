"""
实验配置模块 - 统一管理所有实验参数
"""

import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from datetime import datetime


@dataclass
class ExperimentConfig:
    """实验总配置"""
    
    # ===== 基础配置 =====
    exp_name: str = "default_exp"
    exp_id: Optional[str] = None  # 自动生成
    seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    dtype: str = "float32"  # float32, float16, bfloat16
    
    # ===== 模型配置 =====
    model_name: str = "gpt2"  # gpt2, distilbert, tinyllama, etc.
    model_path: Optional[str] = None  # 预训练模型路径
    use_pretrained: bool = True
    
    # ===== 数据配置 =====
    dataset_name: str = "wikitext"  # wikitext, openwebtext, etc.
    dataset_path: Optional[str] = None
    batch_size: int = 8
    seq_length: int = 128
    num_samples: int = 100  # 实验样本数
    
    # ===== 错误注入配置 =====
    injection_enabled: bool = True
    injection_location: str = "scores"  # q, k, v, scores, weights, out
    injection_stage: str = "inference"  # training, inference, both
    
    # 层选择配置
    injection_layers: Optional[List[int]] = None  # None表示所有层，否则为层索引列表 [0, 1, 2]
    injection_layer_mode: str = "first"  # first, all, random, specific
    
    # 单次注入配置
    injection_idx: tuple = (0, 0, 0, 0)  # (batch, head, i, j)
    injection_bit: int = 15
    
    # 多次注入配置
    multi_injection: bool = False
    num_injections: int = 1
    random_position: bool = False
    random_bit: bool = False
    injection_rate: float = 0.0  # 0表示确定性注入
    
    # ===== 边界检测配置 =====
    bounds_check: bool = True
    tolerance: float = 0.0
    handle_nan: bool = True
    bound_type: str = "s@w" # s@w, q@o, comb
    
    # ===== 实验控制 =====
    num_runs: int = 1  # 重复运行次数
    save_results: bool = True
    save_dir: str = "./results"
    log_interval: int = 10
    
    # ===== 高级配置 =====
    enable_causal_mask: bool = True
    use_kv_cache: bool = False
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if self.exp_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_id = f"{self.exp_name}_{timestamp}"
        
        # 转换injection_idx为tuple
        if isinstance(self.injection_idx, (list, str)):
            if isinstance(self.injection_idx, str):
                self.injection_idx = eval(self.injection_idx)
            self.injection_idx = tuple(self.injection_idx)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """保存配置到文件"""
        if path is None:
            assert self.exp_id is not None, "exp_id must be specified"
            path = Path(self.save_dir) / self.exp_id / "config.json"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """从文件加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def get_device(self) -> torch.device:
        """获取设备"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    def get_dtype(self) -> torch.dtype:
        """获取数据类型"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.dtype, torch.float32)


@dataclass
class ParameterSweepConfig:
    """参数扫描配置"""
    
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)
    
    # 示例:
    # sweep_params = {
    #     'seed': [42, 123, 456],
    #     'injection_bit': list(range(32)),
    #     'injection_location': ['q', 'k', 'v', 'scores', 'weights', 'out']
    # }
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """生成所有参数组合的配置"""
        import itertools
        
        if not self.sweep_params:
            return [self.base_config]
        
        # 获取所有参数名和值列表
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        
        # 生成所有组合
        configs = []
        for values in itertools.product(*param_values):
            # 复制基础配置
            config_dict = self.base_config.to_dict()
            
            # 更新扫描参数
            for name, value in zip(param_names, values):
                config_dict[name] = value
            
            # 创建新配置
            config = ExperimentConfig(**config_dict)
            
            # 更新实验ID以反映参数
            param_str = "_".join([f"{k}={v}" for k, v in zip(param_names, values)])
            config.exp_id = f"{config.exp_name}_{param_str}"
            
            configs.append(config)
        
        return configs
    
    def get_num_configs(self) -> int:
        """获取配置总数"""
        if not self.sweep_params:
            return 1
        
        count = 1
        for values in self.sweep_params.values():
            count *= len(values)
        return count


# 预定义的实验配置模板
class ConfigTemplates:
    """预定义配置模板"""
    
    @staticmethod
    def minimal_test() -> ExperimentConfig:
        """最小测试配置"""
        return ExperimentConfig(
            exp_name="minimal_test",
            model_name="gpt2",
            dataset_name="wikitext",
            batch_size=2,
            seq_length=32,
            num_samples=10,
            num_runs=1
        )
    
    @staticmethod
    def bit_sweep() -> ParameterSweepConfig:
        """比特位扫描配置"""
        base = ExperimentConfig(
            exp_name="bit_sweep",
            model_name="gpt2",
            batch_size=4,
            num_samples=50
        )
        
        return ParameterSweepConfig(
            base_config=base,
            sweep_params={
                'injection_bit': list(range(32)),
                'seed': [42, 123, 456]
            }
        )
    
    @staticmethod
    def location_sweep() -> ParameterSweepConfig:
        """注入位置扫描配置"""
        base = ExperimentConfig(
            exp_name="location_sweep",
            model_name="gpt2",
            batch_size=4,
            num_samples=50
        )
        
        return ParameterSweepConfig(
            base_config=base,
            sweep_params={
                'injection_location': ['q', 'k', 'v', 'scores', 'weights', 'out'],
                'seed': [42, 123, 456]
            }
        )
    
    @staticmethod
    def full_sweep() -> ParameterSweepConfig:
        """完整参数扫描"""
        base = ExperimentConfig(
            exp_name="full_sweep",
            model_name="gpt2",
            batch_size=4,
            num_samples=100
        )
        
        return ParameterSweepConfig(
            base_config=base,
            sweep_params={
                'seed': [42, 123, 456, 789],
                'injection_bit': [0, 7, 15, 23, 31],
                'injection_location': ['scores', 'weights'],
                'injection_stage': ['inference']
            }
        )
