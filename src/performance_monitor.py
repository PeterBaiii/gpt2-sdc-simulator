"""
性能检测模块 - 记录和分析实验各阶段的性能指标
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    
    # 时间指标 (秒)
    total_time: float = 0.0
    baseline_forward_time: float = 0.0
    injection_forward_time: float = 0.0
    bounds_computation_time: float = 0.0
    violation_detection_time: float = 0.0
    
    # 内存指标 (MB)
    peak_memory_allocated: float = 0.0
    peak_memory_reserved: float = 0.0
    baseline_memory: float = 0.0
    injection_memory: float = 0.0
    
    # 计算指标
    num_attention_layers: int = 0
    num_attention_heads: int = 0
    sequence_length: int = 0
    batch_size: int = 0
    
    # FLOPS估算
    attention_flops: float = 0.0
    bounds_flops: float = 0.0
    
    # 额外指标
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'time': {
                'total_time': self.total_time,
                'baseline_forward_time': self.baseline_forward_time,
                'injection_forward_time': self.injection_forward_time,
                'bounds_computation_time': self.bounds_computation_time,
                'violation_detection_time': self.violation_detection_time,
            },
            'memory': {
                'peak_memory_allocated_mb': self.peak_memory_allocated,
                'peak_memory_reserved_mb': self.peak_memory_reserved,
                'baseline_memory_mb': self.baseline_memory,
                'injection_memory_mb': self.injection_memory,
            },
            'compute': {
                'num_attention_layers': self.num_attention_layers,
                'num_attention_heads': self.num_attention_heads,
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'attention_flops': self.attention_flops,
                'bounds_flops': self.bounds_flops,
            },
            'extra': self.extra_metrics
        }
    
    def compute_overhead(self) -> Dict[str, float]:
        """计算开销"""
        baseline_total = self.baseline_forward_time
        injection_total = self.injection_forward_time
        detection_total = self.bounds_computation_time + self.violation_detection_time
        
        overhead = {}
        
        if baseline_total > 0:
            overhead['injection_vs_baseline'] = (injection_total / baseline_total - 1) * 100
            overhead['detection_vs_baseline'] = (detection_total / baseline_total) * 100
            overhead['total_vs_baseline'] = ((injection_total + detection_total) / baseline_total - 1) * 100
        
        return overhead


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = PerformanceMetrics()
        self.timer_stack: List[tuple] = []
        
    def reset(self):
        """重置监控器"""
        self.metrics = PerformanceMetrics()
        self.timer_stack = []
        
    @contextmanager
    def timer(self, name: str):
        """计时器上下文管理器"""
        start_time = time.time()
        
        # GPU同步
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        try:
            yield
        finally:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            
            # 保存到对应字段
            if name == 'baseline_forward':
                self.metrics.baseline_forward_time = elapsed
            elif name == 'injection_forward':
                self.metrics.injection_forward_time = elapsed
            elif name == 'bounds_computation':
                self.metrics.bounds_computation_time = elapsed
            elif name == 'violation_detection':
                self.metrics.violation_detection_time = elapsed
            elif name == 'total':
                self.metrics.total_time = elapsed
            else:
                self.metrics.extra_metrics[f'{name}_time'] = elapsed
    
    def record_memory(self, stage: str = 'current'):
        """记录内存使用"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
            
            self.metrics.peak_memory_allocated = max(
                self.metrics.peak_memory_allocated, allocated
            )
            self.metrics.peak_memory_reserved = max(
                self.metrics.peak_memory_reserved, reserved
            )
            
            if stage == 'baseline':
                self.metrics.baseline_memory = allocated
            elif stage == 'injection':
                self.metrics.injection_memory = allocated
        else:
            # CPU内存
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024**2
            self.metrics.extra_metrics[f'{stage}_cpu_memory_mb'] = mem_mb
    
    def record_model_info(self, model, batch_size: int, seq_length: int):
        """记录模型信息"""
        self.metrics.batch_size = batch_size
        self.metrics.sequence_length = seq_length
        
        # 尝试获取注意力层数和头数
        try:
            if hasattr(model, 'transformer'):
                # GPT-2 style
                self.metrics.num_attention_layers = len(model.transformer.h)
                if hasattr(model.transformer.h[0].attn, 'num_heads'):
                    self.metrics.num_attention_heads = model.transformer.h[0].attn.num_heads
            elif hasattr(model, 'bert'):
                # BERT style
                self.metrics.num_attention_layers = len(model.bert.encoder.layer)
                self.metrics.num_attention_heads = model.bert.encoder.layer[0].attention.self.num_attention_heads
        except:
            pass
    
    def estimate_flops(self, d_model: int, n_heads: int, seq_len: int, batch_size: int):
        """估算FLOPS"""
        # Attention FLOPS: 2 * B * H * T^2 * D (QK^T) + 2 * B * H * T^2 * D (Attn @ V)
        attention_flops = 4 * batch_size * n_heads * seq_len**2 * (d_model // n_heads)
        self.metrics.attention_flops = attention_flops
        
        # Bounds computation FLOPS: O(B * H * T * T) for top-k and comparisons
        bounds_flops = batch_size * n_heads * seq_len * seq_len * 10  # 粗略估计
        self.metrics.bounds_flops = bounds_flops
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        return self.metrics
    
    def print_summary(self):
        """打印性能摘要"""
        m = self.metrics
        overhead = m.compute_overhead()
        
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        
        print("\n[Time Metrics]")
        print(f"  Total time:              {m.total_time:.4f}s")
        print(f"  Baseline forward:        {m.baseline_forward_time:.4f}s")
        print(f"  Injection forward:       {m.injection_forward_time:.4f}s")
        print(f"  Bounds computation:      {m.bounds_computation_time:.4f}s")
        print(f"  Violation detection:     {m.violation_detection_time:.4f}s")
        
        if overhead:
            print("\n[Overhead]")
            for key, value in overhead.items():
                print(f"  {key}: {value:+.2f}%")
        
        print("\n[Memory Metrics]")
        print(f"  Peak allocated:          {m.peak_memory_allocated:.2f} MB")
        print(f"  Peak reserved:           {m.peak_memory_reserved:.2f} MB")
        print(f"  Baseline memory:         {m.baseline_memory:.2f} MB")
        print(f"  Injection memory:        {m.injection_memory:.2f} MB")
        
        print("\n[Model Info]")
        print(f"  Attention layers:        {m.num_attention_layers}")
        print(f"  Attention heads:         {m.num_attention_heads}")
        print(f"  Sequence length:         {m.sequence_length}")
        print(f"  Batch size:              {m.batch_size}")
        
        if m.attention_flops > 0:
            print("\n[FLOPS Estimate]")
            print(f"  Attention:               {m.attention_flops:.2e}")
            print(f"  Bounds computation:      {m.bounds_flops:.2e}")
            if m.baseline_forward_time > 0:
                tflops = m.attention_flops / m.baseline_forward_time / 1e12
                print(f"  Effective TFLOPS:        {tflops:.2f}")
        
        print("="*60 + "\n")


class PerformanceAggregator:
    """性能指标聚合器 - 用于多次实验的统计"""
    
    def __init__(self):
        self.metrics_list: List[PerformanceMetrics] = []
    
    def add(self, metrics: PerformanceMetrics):
        """添加一次实验的指标"""
        self.metrics_list.append(metrics)
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算统计信息"""
        if not self.metrics_list:
            return {}
        
        stats = {}
        
        # 提取所有时间指标
        time_metrics = {
            'total_time': [m.total_time for m in self.metrics_list],
            'baseline_forward_time': [m.baseline_forward_time for m in self.metrics_list],
            'injection_forward_time': [m.injection_forward_time for m in self.metrics_list],
            'bounds_computation_time': [m.bounds_computation_time for m in self.metrics_list],
            'violation_detection_time': [m.violation_detection_time for m in self.metrics_list],
        }
        
        for name, values in time_metrics.items():
            values = [v for v in values if v > 0]  # 过滤掉0值
            if values:
                stats[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # 内存指标
        memory_metrics = {
            'peak_memory_allocated': [m.peak_memory_allocated for m in self.metrics_list],
            'baseline_memory': [m.baseline_memory for m in self.metrics_list],
        }
        
        for name, values in memory_metrics.items():
            values = [v for v in values if v > 0]
            if values:
                stats[name] = {
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values))
                }
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print(f"Performance Statistics ({len(self.metrics_list)} runs)")
        print("="*60)
        
        for metric_name, stat_dict in stats.items():
            print(f"\n{metric_name}:")
            for stat_name, value in stat_dict.items():
                if 'time' in metric_name:
                    print(f"  {stat_name}: {value:.4f}s")
                else:
                    print(f"  {stat_name}: {value:.2f} MB")
        
        print("="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = PerformanceMonitor(device)
    
    # 模拟实验
    with monitor.timer('total'):
        with monitor.timer('baseline_forward'):
            time.sleep(0.1)
        
        monitor.record_memory('baseline')
        
        with monitor.timer('injection_forward'):
            time.sleep(0.12)
        
        monitor.record_memory('injection')
        
        with monitor.timer('bounds_computation'):
            time.sleep(0.05)
        
        with monitor.timer('violation_detection'):
            time.sleep(0.02)
    
    monitor.print_summary()