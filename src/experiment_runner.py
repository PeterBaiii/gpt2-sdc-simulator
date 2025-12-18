"""
实验运行模块 - 统一的实验执行和结果记录
"""

import json
import deprecated
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import pickle
from deprecated import deprecated

@dataclass
class ExperimentResult:
    """单次实验结果"""
    
    # 基础信息
    exp_id: str
    run_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 模型输出
    baseline_loss: Optional[float] = None
    injected_loss: Optional[float] = None
    loss_diff: Optional[float] = None
    
    # 边界检测结果
    bounds_valid: Optional[bool] = None
    violation_detected: Optional[bool] = None
    num_violations: Optional[int] = None
    violation_positions: Optional[List] = None
    
    # 详细统计
    baseline_epsilon_stats: Optional[Dict] = None
    injected_epsilon_stats: Optional[Dict] = None
    
    # 错误注入信息
    injection_info: Optional[Dict] = None
    
    # 额外数据
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'exp_id': self.exp_id,
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'baseline_loss': self.baseline_loss,
            'injected_loss': self.injected_loss,
            'loss_diff': self.loss_diff,
            'bounds_valid': self.bounds_valid,
            'violation_detected': self.violation_detected,
            'num_violations': self.num_violations,
            'violation_positions': self.violation_positions,
            'baseline_epsilon_stats': self.baseline_epsilon_stats,
            'injected_epsilon_stats': self.injected_epsilon_stats,
            'injection_info': self.injection_info,
            'extra_data': self.extra_data
        }
    
    def save(self, path: Union[str, Path]):
        """保存结果"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ResultsLogger:
    """结果记录器"""
    
    def __init__(self, save_dir: str, exp_id: str):
        self.save_dir = Path(save_dir) / exp_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.exp_id = exp_id
        self.results: List[ExperimentResult] = []
        
        # 创建日志文件
        self.log_file = self.save_dir / "experiment.log"
        self.summary_file = self.save_dir / "summary.json"
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def add_result(self, result: ExperimentResult):
        """添加实验结果"""
        self.results.append(result)
        
        # 保存单次结果
        result_file = self.save_dir / f"result_run_{result.run_id}.json"
        result.save(str(result_file))
        
        self.log(f"Result saved for run {result.run_id}")
    
    def compute_summary(self) -> Dict[str, Any]:
        """计算汇总统计"""
        if not self.results:
            return {}
        
        summary = {
            'exp_id': self.exp_id,
            'num_runs': len(self.results),
            'timestamp': datetime.now().isoformat()
        }
        
        # 收集所有指标
        baseline_losses = [r.baseline_loss for r in self.results if r.baseline_loss is not None]
        injected_losses = [r.injected_loss for r in self.results if r.injected_loss is not None]
        loss_diffs = [r.loss_diff for r in self.results if r.loss_diff is not None]
        
        violations = [r.violation_detected for r in self.results if r.violation_detected is not None]
        num_violations = [r.num_violations for r in self.results if r.num_violations is not None]
        
        # 统计
        if baseline_losses:
            summary['baseline_loss'] = {
                'mean': float(np.mean(baseline_losses)),
                'std': float(np.std(baseline_losses)),
                'min': float(np.min(baseline_losses)),
                'max': float(np.max(baseline_losses))
            }
        
        if injected_losses:
            summary['injected_loss'] = {
                'mean': float(np.mean(injected_losses)),
                'std': float(np.std(injected_losses)),
                'min': float(np.min(injected_losses)),
                'max': float(np.max(injected_losses))
            }
        
        if loss_diffs:
            summary['loss_diff'] = {
                'mean': float(np.mean(loss_diffs)),
                'std': float(np.std(loss_diffs)),
                'min': float(np.min(loss_diffs)),
                'max': float(np.max(loss_diffs))
            }
        
        if violations:
            summary['violation_detection'] = {
                'detection_rate': float(np.mean(violations)),
                'total_violations': sum(num_violations) if num_violations else 0,
                'avg_violations_per_run': float(np.mean(num_violations)) if num_violations else 0
            }
        
        return summary
    
    def save_summary(self):
        """保存汇总结果"""
        summary = self.compute_summary()
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log("Summary saved")
        return summary
    
    def save_all_results(self):
        """保存所有结果到单个文件"""
        all_results = [r.to_dict() for r in self.results]
        
        results_file = self.save_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # 也保存为pickle格式便于后续分析
        pickle_file = self.save_dir / "all_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        self.log(f"All results saved ({len(self.results)} runs)")

class ViolationLogger:
    """专门记录违背边界的配置和详细输出"""
    
    def __init__(self, save_dir: str, exp_name: str):
        """
        Args:
            save_dir: 保存目录
            exp_name: 实验名称
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建违背日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.save_dir / f"violations_{exp_name}_{timestamp}.log"
        self.json_file = self.save_dir / f"violations_{exp_name}_{timestamp}.json"
        
        self.violations = []
        
        # 初始化文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Violation Log for Experiment: {exp_name}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log_violation(self, config_idx: int, config: Any, result: Any):
        """
        记录一个违背配置
        
        Args:
            config_idx: 配置索引
            config: 配置对象
            result: 实验结果对象
        """
        violation_entry = {
            'config_idx': config_idx,
            'timestamp': datetime.now().isoformat(),
            'config': self._extract_config_info(config),
            'result': self._extract_result_info(result)
        }
        
        self.violations.append(violation_entry)
        
        # 写入文本日志
        self._write_text_log(violation_entry)
        
        # 更新JSON文件
        self._write_json_log()
    
    def _extract_config_info(self, config) -> Dict[str, Any]:
        """提取配置信息"""
        return {
            'seed': config.seed,
            'injection_layers': config.injection_layers,
            'injection_bit': config.injection_bit,
            'injection_location': config.injection_location,
            'injection_idx': config.injection_idx,
            'model_name': config.model_name,
            'batch_size': config.batch_size,
            'seq_length': config.seq_length,
        }
    
    def _extract_result_info(self, result) -> Dict[str, Any]:
        """提取结果信息"""
        info = {
            'baseline_loss': result.baseline_loss,
            'injected_loss': result.injected_loss,
            'loss_diff': result.loss_diff,
            'violation_detected': result.violation_detected,
            'num_violations': result.num_violations,
            'violation_positions': result.violation_positions,
        }
        
        # 提取epsilon分析
        if 'epsilon_analysis' in result.extra_data:
            info['epsilon_analysis'] = []
            for layer_analysis in result.extra_data['epsilon_analysis']:
                layer_info = {
                    'layer_idx': layer_analysis['layer_idx'],
                    'mean_diff': layer_analysis['analysis']['mean_diff'],
                    'std_diff': layer_analysis['analysis']['std_diff'],
                    'max_abs_diff': layer_analysis['analysis']['max_abs_diff'],
                    'top_changes': []
                }
                
                # 保存top 5
                for pos, changes, bounds in layer_analysis['analysis']['top_changes'][:5]:
                    layer_info['top_changes'].append({
                        'position': pos,
                        'baseline_epsilon': changes['baseline_epsilon'],
                        'injected_epsilon': changes['injected_epsilon'],
                        'epsilon_diff': changes['epsilon_diff'],
                        'abs_diff': changes['abs_diff'],
                        'middle_bound': bounds['middle'],
                        'upper_bound': bounds['upper'],
                        'gamma': bounds['gamma'],
                    })
                
                info['epsilon_analysis'].append(layer_info)
        
        return info
    
    def _write_text_log(self, entry: Dict[str, Any]):
        """写入文本日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"VIOLATION DETECTED - Config #{entry['config_idx']}\n")
            f.write(f"Time: {entry['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            # 配置信息
            f.write("Configuration:\n")
            f.write("-"*80 + "\n")
            config = entry['config']
            f.write(f"  Seed:              {config['seed']}\n")
            f.write(f"  Injection Layers:  {config['injection_layers']}\n")
            f.write(f"  Bit Position:      {config['injection_bit']}\n")
            f.write(f"  Tensor Type:       {config['injection_location']}\n")
            f.write(f"  Spatial Position:  {config['injection_idx']}\n")
            f.write(f"  Model:             {config['model_name']}\n")
            f.write(f"  Batch Size:        {config['batch_size']}\n")
            f.write(f"  Sequence Length:   {config['seq_length']}\n")
            f.write("\n")
            
            # 结果信息
            result = entry['result']
            f.write("Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Baseline Loss:      {result['baseline_loss']:.6f}\n")
            f.write(f"  Injected Loss:      {result['injected_loss']:.6f}\n")
            f.write(f"  Loss Difference:    {result['loss_diff']:.6f}\n")
            f.write(f"  Num Violations:     {result['num_violations']}\n")
            
            if result['violation_positions']:
                f.write(f"\n  Violation Positions (first 5):\n")
                for pos in result['violation_positions'][:5]:
                    f.write(f"    {pos}\n")
            
            # Epsilon分析
            if 'epsilon_analysis' in result:
                f.write("\n")
                f.write("Epsilon Analysis:\n")
                f.write("-"*80 + "\n")
                
                for layer_info in result['epsilon_analysis']:
                    layer_idx = layer_info['layer_idx']
                    f.write(f"\n  Layer {layer_idx}:\n")
                    f.write(f"    Mean epsilon diff:    {layer_info['mean_diff']:.6f}\n")
                    f.write(f"    Std epsilon diff:     {layer_info['std_diff']:.6f}\n")
                    f.write(f"    Max |epsilon diff|:   {layer_info['max_abs_diff']:.6f}\n")
                    
                    if layer_info['top_changes']:
                        f.write(f"\n    Top 5 Epsilon Changes:\n")
                        for i, change in enumerate(layer_info['top_changes'], 1):
                            f.write(f"      #{i} Position {change['position']}:\n")
                            f.write(f"        Baseline ε:  {change['baseline_epsilon']:.6f}\n")
                            f.write(f"        Injected ε:  {change['injected_epsilon']:.6f}\n")
                            f.write(f"        Δε:          {change['epsilon_diff']:.6f}\n")
                            f.write(f"        Bounds:      [{change['middle_bound']:.6f}, {change['upper_bound']:.6f}]\n")
                            f.write(f"        γ (margin):  {change['gamma']:.6f}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    def _write_json_log(self):
        """写入JSON格式的汇总"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_violations': len(self.violations),
                'violations': self.violations
            }, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> str:
        """获取汇总信息"""
        if not self.violations:
            return "No violations detected."
        
        summary = f"\nViolation Summary:\n"
        summary += f"  Total violations detected: {len(self.violations)}\n"
        summary += f"  Log file: {self.log_file}\n"
        summary += f"  JSON file: {self.json_file}\n"
        
        return summary
    
# @deprecated("")
# class ExperimentRunner:
#     """实验运行器"""
    
#     def __init__(self, config):
#         """
#         Args:
#             config: ExperimentConfig对象
#         """
#         self.config = config
#         self.logger = ResultsLogger(config.save_dir, config.exp_id)
        
#         # 设置随机种子
#         self._set_seed(config.seed)
        
#         self.device = config.get_device()
#         self.dtype = config.get_dtype()
        
#         self.logger.log(f"Experiment initialized: {config.exp_id}")
#         self.logger.log(f"Device: {self.device}, Dtype: {self.dtype}")
    
#     def _set_seed(self, seed: int):
#         """设置随机种子"""
#         import random
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
    
#     def run_single(self, run_id: int, model, dataloader, 
#                    return_tensors: bool = False) -> ExperimentResult:
#         """
#         运行单次实验
        
#         Args:
#             run_id: 运行ID
#             model: 模型对象(已经过monkey patch)
#             dataloader: 数据加载器
#             return_tensors: 是否返回中间张量(占用大量内存)
            
#         Returns:
#             ExperimentResult对象
#         """
#         from performance_monitor import PerformanceMonitor
#         from fault_injection import InjectionConfig, InjectionLocation
#         from bounds_computation import compute_attention_bounds, detect_violation, compute_injected_epsilon
        
#         self.logger.log(f"Starting run {run_id}")
        
#         # 初始化性能监控
#         monitor = PerformanceMonitor(self.device)
        
#         result = ExperimentResult(
#             exp_id=self.config.exp_id,
#             run_id=run_id
#         )
        
#         # 获取一个batch的数据
#         try:
#             batch = next(iter(dataloader))
#         except StopIteration:
#             self.logger.log("No data in dataloader!", level="ERROR")
#             return result
        
#         # 移动到设备
#         if isinstance(batch, dict):
#             batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
#                     for k, v in batch.items()}
#             input_ids = batch.get('input_ids')
#             attention_mask = batch.get('attention_mask', None)
#         else:
#             input_ids = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
#             attention_mask = None
        
#         assert input_ids is not None, "input_ids not found in batch"
#         batch_size, seq_length = input_ids.shape
        
#         # 记录模型信息
#         monitor.record_model_info(model, batch_size, seq_length)
        
#         with monitor.timer('total'):
#             # ==================== Baseline运行 ====================
#             self.logger.log("Running baseline (no injection)...")
#             monitor.record_memory('before_baseline')
            
#             model.eval()
#             baseline_intermediates = {}
            
#             with torch.no_grad():
#                 with monitor.timer('baseline_forward'):
#                     # 运行baseline
#                     if hasattr(model, 'transformer'):  # GPT-2 style
#                         # 需要手动forward以捕获中间张量
#                         outputs = model(input_ids, attention_mask=attention_mask, 
#                                       output_hidden_states=True, return_dict=True)
                        
#                         # 收集所有层的中间张量
#                         for layer_idx, layer in enumerate(model.transformer.h):
#                             if hasattr(layer.attn, 'adapter'):
#                                 # 这一层被patch了，可以获取中间张量
#                                 # 但baseline运行时没有注入，需要重新forward一次该层
#                                 pass
#                     else:
#                         outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
                    
#                     if hasattr(outputs, 'loss') and outputs.loss is not None:
#                         baseline_loss = outputs.loss.item()
#                     else:
#                         baseline_loss = None
            
#             monitor.record_memory('baseline')
#             result.baseline_loss = baseline_loss
#             self.logger.log(f"Baseline loss: {baseline_loss}")
            
#             # ==================== 收集baseline的中间张量 ====================
#             self.logger.log("Collecting baseline intermediates...")
            
#             # 重新运行一次以收集中间张量
#             # 这次我们需要从patch的层中获取q,k,v,scores,weights
#             collected_layers = {}
            
#             with torch.no_grad():
#                 # 临时修改模型以返回中间张量
#                 for layer_idx, layer in enumerate(model.transformer.h):
#                     if hasattr(layer.attn, 'adapter') and hasattr(layer.attn, 'layer_idx'):
#                         # 标记这一层需要返回中间张量
#                         layer.attn._return_intermediates = True
                
#                 # Forward一次
#                 _ = model(input_ids, attention_mask=attention_mask, return_dict=True)
                
#                 # 收集中间张量 (需要在adapter中实现保存机制)
#                 # 这里假设我们通过某种方式能够获取
#                 # TODO: 实现更优雅的中间张量收集机制
            
#             # ==================== 注入错误运行 ====================
#             if self.config.injection_enabled:
#                 self.logger.log("Running with fault injection...")
#                 monitor.record_memory('before_injection')
                
#                 # 创建注入配置
#                 injection_cfg = InjectionConfig(
#                     location=InjectionLocation(self.config.injection_location),
#                     idx=self.config.injection_idx,
#                     bit=self.config.injection_bit,
#                     enabled=True,
#                     random_position=self.config.random_position,
#                     injection_rate=self.config.injection_rate
#                 )
                
#                 result.injection_info = injection_cfg.to_dict()
                
#                 injected_intermediates = {}
                
#                 with torch.no_grad():
#                     with monitor.timer('injection_forward'):
#                         # TODO: 实现注入版本的forward
#                         # 需要将injection_cfg传递给adapter
                        
#                         outputs_inj = model(input_ids, attention_mask=attention_mask, 
#                                           return_dict=True)
                        
#                         if hasattr(outputs_inj, 'loss') and outputs_inj.loss is not None:
#                             injected_loss = outputs_inj.loss.item()
#                         else:
#                             injected_loss = None
                
#                 monitor.record_memory('injection')
#                 result.injected_loss = injected_loss
                
#                 if baseline_loss is not None and injected_loss is not None:
#                     result.loss_diff = injected_loss - baseline_loss
                
#                 self.logger.log(f"Injected loss: {injected_loss}")
#                 self.logger.log(f"Loss diff: {result.loss_diff}")
                
#                 # ==================== 计算边界 ====================
#                 self.logger.log("Computing bounds...")
                
#                 with monitor.timer('bounds_computation'):
#                     # 对每个被patch的层计算边界
#                     bounds_results = {}
                    
#                     # TODO: 从collected_layers中获取scores和weights
#                     # 这里需要实现具体的张量提取逻辑
                    
#                     # 示例代码(假设我们有了scores和p)
#                     # for layer_idx in collected_layers:
#                     #     scores = collected_layers[layer_idx]['scores']
#                     #     p = collected_layers[layer_idx]['weights']
#                     #     d = collected_layers[layer_idx]['head_dim']
#                     #     
#                     #     bounds = compute_attention_bounds(scores, p, d)
#                     #     bounds_results[layer_idx] = bounds
                
#                 # ==================== 检测违反 ====================
#                 with monitor.timer('violation_detection'):
#                     # TODO: 实现违反检测逻辑
#                     violation_detected = False
#                     num_violations = 0
                    
#                     # for layer_idx, bounds in bounds_results.items():
#                     #     # 计算注入后的epsilon
#                     #     injected_eps = compute_injected_epsilon(...)
#                     #     
#                     #     # 检测违反
#                     #     detection = detect_violation(bounds, injected_eps, self.config.tolerance)
#                     #     
#                     #     if detection['injection_violations']['any_violated']:
#                     #         violation_detected = True
#                     #         num_violations += detection['injection_violations']['num_upper_violations']
                
#                 result.violation_detected = violation_detected
#                 result.num_violations = num_violations
                
#                 self.logger.log(f"Violations detected: {violation_detected} ({num_violations} total)")
        
#         # 性能统计
#         perf_metrics = monitor.get_metrics()
#         result.extra_data['performance'] = perf_metrics.to_dict()
        
#         monitor.print_summary()
        
#         self.logger.log(f"Run {run_id} completed")
        
#         return result
    
#     def run_all(self, model, dataloader) -> List[ExperimentResult]:
#         """
#         运行所有重复实验
        
#         Args:
#             model: 模型对象
#             dataloader: 数据加载器
            
#         Returns:
#             结果列表
#         """
#         self.logger.log(f"Starting experiment with {self.config.num_runs} runs")
        
#         results = []
#         for run_id in range(self.config.num_runs):
#             result = self.run_single(run_id, model, dataloader)
#             self.logger.add_result(result)
#             results.append(result)
        
#         # 保存汇总
#         summary = self.logger.save_summary()
#         self.logger.save_all_results()
        
#         self.logger.log("Experiment completed")
#         self.logger.log(f"Summary: {json.dumps(summary, indent=2)}")
        
#         return results

# @deprecated("")
# def run_parameter_sweep(sweep_config, model_fn, data_fn):
#     """
#     运行参数扫描实验
    
#     Args:
#         sweep_config: ParameterSweepConfig对象
#         model_fn: 模型创建函数 (config) -> model
#         data_fn: 数据创建函数 (config) -> dataloader
        
#     Returns:
#         所有结果的列表
#     """
#     configs = sweep_config.generate_configs()
#     print(f"Running parameter sweep with {len(configs)} configurations")
    
#     all_results = []
    
#     for i, config in enumerate(configs):
#         print(f"\n{'='*60}")
#         print(f"Configuration {i+1}/{len(configs)}: {config.exp_id}")
#         print(f"{'='*60}\n")
        
#         # 创建模型和数据
#         model = model_fn(config)
#         dataloader = data_fn(config)
        
#         # 运行实验
#         runner = ExperimentRunner(config)
#         results = runner.run_all(model, dataloader)
        
#         all_results.extend(results)
    
#     print(f"\nParameter sweep completed: {len(all_results)} total runs")
    
#     return all_results