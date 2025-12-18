"""
错误注入模块 - 提供多种错误注入策略
"""

import torch
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class InjectionLocation(Enum):
    """错误注入位置枚举"""
    Q = "q"
    K = "k"
    V = "v"
    SCORES = "scores"
    WEIGHTS = "weights"
    OUT = "out"
    NONE = "none"


@dataclass
class InjectionConfig:
    """错误注入配置"""
    location: InjectionLocation = InjectionLocation.NONE
    idx: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bit: int = 0
    enabled: bool = False
    
    # 扩展配置
    multi_bit: bool = False  # 是否多比特翻转
    bit_list: Optional[list] = None  # 多比特翻转的比特列表
    random_position: bool = False  # 是否随机选择位置
    injection_rate: float = 0.0  # 注入概率(0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典,便于记录"""
        return {
            'location': self.location.value,
            'idx': self.idx,
            'bit': self.bit,
            'enabled': self.enabled,
            'multi_bit': self.multi_bit,
            'bit_list': self.bit_list,
            'random_position': self.random_position,
            'injection_rate': self.injection_rate
        }


@torch.no_grad()
def bitflip_(t: torch.Tensor, idx: tuple, bit: int) -> None:
    """
    原地比特翻转 (支持 fp16/bf16/fp32)
    
    Args:
        t: 目标张量
        idx: 索引位置
        bit: 要翻转的比特位
    """
    assert t.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"Unsupported dtype: {t.dtype}"
    
    if t.dtype == torch.float32:
        iview = t.view(torch.int32)
        bit = bit & 31
    else:
        iview = t.view(torch.int16)
        bit = bit & 15
    
    iview[idx] ^= (1 << bit)


@torch.no_grad()
def multi_bitflip_(t: torch.Tensor, idx: tuple, bits: list) -> None:
    """
    多比特翻转
    
    Args:
        t: 目标张量
        idx: 索引位置
        bits: 要翻转的比特位列表
    """
    for bit in bits:
        bitflip_(t, idx, bit)


@torch.no_grad()
def random_bitflip_(t: torch.Tensor, num_flips: int = 1, 
                    bit_range: Optional[Tuple[int, int]] = None) -> list:
    """
    随机位置的比特翻转
    
    Args:
        t: 目标张量
        num_flips: 翻转次数
        bit_range: 比特范围 (min_bit, max_bit)
        
    Returns:
        injection_records: 注入记录列表
    """
    import random
    
    shape = t.shape
    max_bit = 31 if t.dtype == torch.float32 else 15
    
    if bit_range is None:
        bit_range = (0, max_bit)
    
    records = []
    for _ in range(num_flips):
        # 随机选择位置
        idx = tuple(random.randint(0, s-1) for s in shape)
        # 随机选择比特位
        bit = random.randint(bit_range[0], bit_range[1])
        
        bitflip_(t, idx, bit)
        records.append({'idx': idx, 'bit': bit})
    
    return records


class FaultInjector:
    """错误注入器类"""
    
    def __init__(self, config: InjectionConfig):
        self.config = config
        self.injection_count = 0
        self.injection_history = []
    
    def inject(self, tensor: torch.Tensor, 
               force: bool = False) -> Optional[Dict[str, Any]]:
        """
        执行错误注入
        
        Args:
            tensor: 目标张量
            force: 是否强制注入(忽略injection_rate)
            
        Returns:
            injection_info: 注入信息(如果执行了注入)
        """
        if not self.config.enabled and not force:
            return None
        
        # 概率性注入
        if not force and self.config.injection_rate > 0:
            import random
            if random.random() > self.config.injection_rate:
                return None
        
        info = {
            'count': self.injection_count,
            'location': self.config.location.value,
            'tensor_shape': tuple(tensor.shape)
        }
        
        if self.config.random_position:
            # 随机位置注入
            records = random_bitflip_(tensor, num_flips=1)
            info.update(records[0])
        elif self.config.multi_bit:
            # 多比特注入
            bits = self.config.bit_list or [self.config.bit]
            multi_bitflip_(tensor, self.config.idx, bits)
            info['idx'] = self.config.idx
            info['bits'] = bits
        else:
            # 单比特注入
            bitflip_(tensor, self.config.idx, self.config.bit)
            info['idx'] = self.config.idx
            info['bit'] = self.config.bit
        
        self.injection_count += 1
        self.injection_history.append(info)
        
        return info
    
    def reset(self):
        """重置注入器状态"""
        self.injection_count = 0
        self.injection_history = []
    
    def get_history(self) -> list:
        """获取注入历史"""
        return self.injection_history.copy()