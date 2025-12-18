"""
实验记录模块 - 统一的实验日志和结果记录
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