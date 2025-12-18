"""
结果分析工具 - 分析实验JSON结果
"""

import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd


class ResultsAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 结果目录路径
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.all_results = []
        
    def parse_config_from_exp_id(self, exp_id: str) -> Optional[Dict]:
        """从实验ID中解析配置参数
        
        exp_id格式: {exp_name}_{key1}={value1}_{key2}={value2}_...
        例如: gpt2_comprehensive_sweep_seed=42_injection_layers=[0]_injection_bit=0
        
        关键: 键名可能包含下划线(如injection_layers)，需要智能解析
        """
        config = {}
        
        # 找到第一个=的位置，确定配置开始的地方
        first_eq = exp_id.find('=')
        if first_eq == -1:
            return None
        
        # 向前找到最近的_（这个_之前是exp_name，之后才是配置）
        last_sep_before_first_eq = exp_id.rfind('_', 0, first_eq)
        
        # 从配置开始的位置解析
        if last_sep_before_first_eq != -1:
            config_str = exp_id[last_sep_before_first_eq + 1:]
        else:
            config_str = exp_id
        
        # 逐个解析键值对
        i = 0
        while i < len(config_str):
            # 找到下一个=
            eq_pos = config_str.find('=', i)
            if eq_pos == -1:
                break
            
            # 键名：从当前位置到=之前
            # 如果i>0，需要跳过前导的_
            if i > 0 and config_str[i] == '_':
                key = config_str[i+1:eq_pos]
                key_actual_start = i + 1
            else:
                key = config_str[i:eq_pos]
                key_actual_start = i
            
            # 向后找值的结束
            value_start = eq_pos + 1
            
            # 如果值是列表或元组，找到匹配的括号
            if value_start < len(config_str) and config_str[value_start] in '[(':
                bracket = config_str[value_start]
                close_bracket = ']' if bracket == '[' else ')'
                depth = 1
                value_end = value_start + 1
                
                while value_end < len(config_str) and depth > 0:
                    if config_str[value_end] == bracket:
                        depth += 1
                    elif config_str[value_end] == close_bracket:
                        depth -= 1
                    value_end += 1
            else:
                # 普通值，找到下一个_之后跟着key=的位置
                value_end = value_start
                while value_end < len(config_str):
                    if config_str[value_end] == '_':
                        # 检查_后面是否有=（表示下一个键值对）
                        rest = config_str[value_end+1:]
                        if '=' in rest:
                            break
                    value_end += 1
            
            value = config_str[value_start:value_end]
            
            # 解析值
            if value.startswith('[') and value.endswith(']'):
                try:
                    config[key] = eval(value)
                except:
                    config[key] = value
            elif value.startswith('(') and value.endswith(')'):
                try:
                    config[key] = eval(value)
                except:
                    config[key] = value
            else:
                try:
                    config[key] = int(value)
                except ValueError:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
            
            # 移动到下一个键值对
            i = value_end
        
        return config if config else None
    
    def load_experiment(self, exp_id: str, parent_dir: Optional[Path] = None):
        """加载单个实验的结果"""
        # 如果指定了parent_dir，则在parent_dir下查找
        if parent_dir:
            exp_dir = parent_dir / exp_id
        else:
            exp_dir = self.results_dir / exp_id
        
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            return None
        
        # 加载配置
        config_file = exp_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # 尝试从exp_id解析配置
            config = self.parse_config_from_exp_id(exp_id)
            if not config:
                config = None
        
        # 加载所有结果
        all_results_file = exp_dir / "all_results.json"
        if all_results_file.exists():
            with open(all_results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        # 加载摘要
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = None
        
        self.experiments[exp_id] = {
            'config': config,
            'results': results,
            'summary': summary,
            'exp_dir': exp_dir
        }
        
        return self.experiments[exp_id]
    
    def load_all_experiments(self):
        """加载所有实验"""
        exp_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        print(f"Found {len(exp_dirs)} experiment directories")
        
        for exp_dir in exp_dirs:
            exp_id = exp_dir.name
            self.load_experiment(exp_id)
        
        print(f"Loaded {len(self.experiments)} experiments")
    
    def analyze_single_experiment(self, exp_id: str):
        """分析单个实验"""
        if exp_id not in self.experiments:
            self.load_experiment(exp_id)
        
        exp = self.experiments[exp_id]
        
        print("="*80)
        print(f"Experiment Analysis: {exp_id}")
        print("="*80)
        
        # 1. 配置信息
        if exp['config']:
            print("\n1. Configuration:")
            print("-"*80)
            config = exp['config']
            print(f"  Model:              {config.get('model_name')}")
            print(f"  Dataset:            {config.get('dataset_name')}")
            print(f"  Batch size:         {config.get('batch_size')}")
            print(f"  Sequence length:    {config.get('seq_length')}")
            print(f"  Injection enabled:  {config.get('injection_enabled')}")
            if config.get('injection_enabled'):
                print(f"  Injection location: {config.get('injection_location')}")
                print(f"  Injection layers:   {config.get('injection_layers')}")
                print(f"  Injection bit:      {config.get('injection_bit')}")
                print(f"  Injection idx:      {config.get('injection_idx')}")
            print(f"  Num runs:           {config.get('num_runs')}")
            print(f"  Seed:               {config.get('seed')}")
        
        # 2. 汇总统计
        if exp['summary']:
            print("\n2. Summary Statistics:")
            print("-"*80)
            summary = exp['summary']
            
            if 'baseline_loss' in summary:
                bl = summary['baseline_loss']
                print(f"  Baseline Loss:")
                print(f"    Mean: {bl['mean']:.6f} ± {bl['std']:.6f}")
                print(f"    Range: [{bl['min']:.6f}, {bl['max']:.6f}]")
            
            if 'injected_loss' in summary:
                il = summary['injected_loss']
                print(f"  Injected Loss:")
                print(f"    Mean: {il['mean']:.6f} ± {il['std']:.6f}")
                print(f"    Range: [{il['min']:.6f}, {il['max']:.6f}]")
            
            if 'loss_diff' in summary:
                ld = summary['loss_diff']
                print(f"  Loss Difference:")
                print(f"    Mean: {ld['mean']:.6f} ± {ld['std']:.6f}")
                print(f"    Range: [{ld['min']:.6f}, {ld['max']:.6f}]")
            
            if 'violation_detection' in summary:
                vd = summary['violation_detection']
                print(f"  Violation Detection:")
                print(f"    Detection rate:      {vd['detection_rate']*100:.1f}%")
                print(f"    Total violations:    {vd['total_violations']}")
                print(f"    Avg per run:         {vd['avg_violations_per_run']:.1f}")
        
        results = {}
        # 3. 详细结果分析
        if exp['results']:
            print("\n3. Detailed Results Analysis:")
            print("-"*80)
            
            results = exp['results']
            num_runs = len(results)
            print(f"  Total runs: {num_runs}")
            
            # 统计违反情况
            violations = [r for r in results if r.get('violation_detected')]
            print(f"  Runs with violations: {len(violations)} ({100*len(violations)/num_runs:.1f}%)")
            
            # Loss分析
            baseline_losses = [r['baseline_loss'] for r in results if r.get('baseline_loss') is not None]
            injected_losses = [r['injected_loss'] for r in results if r.get('injected_loss') is not None]
            loss_diffs = [r['loss_diff'] for r in results if r.get('loss_diff') is not None]
            
            if loss_diffs:
                print(f"\n  Loss Difference Statistics:")
                print(f"    Mean:   {np.mean(loss_diffs):.6f}")
                print(f"    Std:    {np.std(loss_diffs):.6f}")
                print(f"    Median: {np.median(loss_diffs):.6f}")
                print(f"    Max:    {np.max(np.abs(loss_diffs)):.6f}")
            
            # Epsilon分析
            print(f"\n  Epsilon Analysis:")
            all_epsilon_changes = []
            
            for r in results:
                if 'extra_data' in r and 'epsilon_analysis' in r['extra_data']:
                    for layer_analysis in r['extra_data']['epsilon_analysis']:
                        analysis = layer_analysis['analysis']
                        all_epsilon_changes.append({
                            'layer': layer_analysis['layer_idx'],
                            'mean_diff': analysis['mean_diff'],
                            'std_diff': analysis['std_diff'],
                            'max_abs_diff': analysis['max_abs_diff']
                        })
            
            if all_epsilon_changes:
                print(f"    Total analyzed: {len(all_epsilon_changes)} layers")
                mean_diffs = [e['mean_diff'] for e in all_epsilon_changes]
                max_diffs = [e['max_abs_diff'] for e in all_epsilon_changes]
                print(f"    Avg mean epsilon diff:  {np.mean(mean_diffs):.6f}")
                print(f"    Avg max |epsilon diff|: {np.mean(max_diffs):.6f}")
                print(f"    Overall max diff:       {np.max(max_diffs):.6f}")
        
        # 4. 性能指标
        if exp['results']:
            print("\n4. Performance Metrics:")
            print("-"*80)
            
            perf_metrics = []
            for r in results:
                if 'extra_data' in r and 'performance' in r['extra_data']:
                    perf_metrics.append(r['extra_data']['performance'])
            
            if perf_metrics:
                # 时间指标
                total_times = [p['time']['total_time'] for p in perf_metrics]
                baseline_times = [p['time']['baseline_forward_time'] for p in perf_metrics]
                injection_times = [p['time']['injection_forward_time'] for p in perf_metrics]
                
                print(f"  Average Times:")
                print(f"    Total:              {np.mean(total_times):.4f}s")
                print(f"    Baseline forward:   {np.mean(baseline_times):.4f}s")
                print(f"    Injection forward:  {np.mean(injection_times):.4f}s")
                
                # 内存指标
                peak_mems = [p['memory']['peak_memory_allocated_mb'] for p in perf_metrics]
                print(f"  Memory:")
                print(f"    Peak allocated:     {np.mean(peak_mems):.2f} MB")
        
        print("\n" + "="*80 + "\n")
    
    def compare_experiments(self, exp_ids: List[str]):
        """对比多个实验"""
        print("="*80)
        print(f"Comparing {len(exp_ids)} Experiments")
        print("="*80)
        
        # 加载所有实验
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                self.load_experiment(exp_id)
        
        # 创建对比表格
        comparison_data = []
        
        for exp_id in exp_ids:
            exp = self.experiments.get(exp_id)
            if not exp:
                continue
            
            row = {'exp_id': exp_id}
            
            # 配置
            if exp['config']:
                config = exp['config']
                row['injection_location'] = config.get('injection_location')
                row['injection_bit'] = config.get('injection_bit')
                row['injection_layers'] = str(config.get('injection_layers'))
                row['seed'] = config.get('seed')
            
            # 结果
            if exp['summary']:
                summary = exp['summary']
                if 'loss_diff' in summary:
                    row['mean_loss_diff'] = summary['loss_diff']['mean']
                    row['std_loss_diff'] = summary['loss_diff']['std']
                if 'violation_detection' in summary:
                    row['detection_rate'] = summary['violation_detection']['detection_rate']
                    row['total_violations'] = summary['violation_detection']['total_violations']
            
            comparison_data.append(row)
        
        # 显示对比
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string())
        print("\n" + "="*80 + "\n")
        
        return df
    
    def analyze_parameter_sweep(self, sweep_dir: str):
        """分析参数扫描结果"""
        sweep_path = self.results_dir / sweep_dir
        
        if not sweep_path.exists():
            print(f"Sweep directory not found: {sweep_path}")
            return
        
        print("="*80)
        print(f"Parameter Sweep Analysis: {sweep_dir}")
        print("="*80)
        
        # 加载violations.json（如果存在）
        violations_dir = sweep_path / "violations"
        violations_data = None
        
        if violations_dir.exists():
            # 查找violations_*.json文件
            violation_files = list(violations_dir.glob("violations_*.json"))
            if violation_files:
                print(f"\nFound violations file: {violation_files[0].name}")
                with open(violation_files[0], 'r') as f:
                    violations_data = json.load(f)
                print(f"Total violations logged: {violations_data.get('total_violations', 0)}")
        
        # 收集所有实验目录
        sweep_experiments = []
        for exp_dir in sweep_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'violations':
                exp_id = exp_dir.name
                # 使用parent_dir参数指定在sweep_path下查找
                if exp_id not in self.experiments:
                    self.load_experiment(exp_id, parent_dir=sweep_path)
                if exp_id in self.experiments:
                    sweep_experiments.append(exp_id)
        
        print(f"\nFound {len(sweep_experiments)} configurations")
        
        # 按参数分组分析
        by_bit = defaultdict(list)
        by_location = defaultdict(list)
        by_layer = defaultdict(list)
        by_seed = defaultdict(list)
        by_idx = defaultdict(list)
        
        # 记录NaN情况
        total_nan_count = 0
        nan_in_violations = []  # 只记录有violation且有NaN的实验
        
        for exp_id in sweep_experiments:
            exp = self.experiments[exp_id]
            
            # 优先使用config，如果没有则尝试解析exp_id
            config = exp.get('config')
            if not config:
                config = self.parse_config_from_exp_id(exp_id)
            
            if not config:
                print(f"Warning: No config found for {exp_id}")
                continue
            
            summary = exp.get('summary')
            
            # 提取关键指标
            detection_rate = 0
            mean_loss_diff = 0
            has_nan = False
            has_violation = False
            
            if summary:
                if 'violation_detection' in summary:
                    detection_rate = summary['violation_detection'].get('detection_rate', 0)
                    total_violations = summary['violation_detection'].get('total_violations', 0)
                    if total_violations > 0:
                        has_violation = True
                        
                if 'loss_diff' in summary:
                    mean_loss_diff = summary['loss_diff'].get('mean', 0)
                    # 检查是否为NaN
                    if np.isnan(mean_loss_diff):
                        has_nan = True
                        mean_loss_diff = 0  # 用0替代NaN进行后续处理
            else:
                # 如果没有summary，尝试从results计算
                results = exp.get('results', [])
                if results:
                    violations = [r.get('violation_detected', False) for r in results if isinstance(r, dict)]
                    if violations:
                        detection_rate = sum(violations) / len(violations)
                        if any(violations):
                            has_violation = True
                    
                    # 过滤掉NaN值
                    loss_diffs = []
                    nan_found_in_results = False
                    for r in results:
                        if isinstance(r, dict) and r.get('loss_diff') is not None:
                            raw = r.get('loss_diff')
                            if raw is None:
                                continue
                            try:
                                diff = float(raw)
                            except Exception:
                                continue
                            
                            if np.isnan(diff):
                                nan_found_in_results = True
                            else:
                                loss_diffs.append(diff)
                    
                    if nan_found_in_results:
                        has_nan = True
                    
                    if loss_diffs:
                        mean_loss_diff = np.mean(loss_diffs)
                    else:
                        mean_loss_diff = 0
            
            # 统计NaN（每个实验只计数一次）
            if has_nan:
                total_nan_count += 1
            
            # 记录既有violation又有NaN的实验
            if has_nan and has_violation:
                nan_in_violations.append({
                    'exp_id': exp_id,
                    'config': config
                })
            
            # 提取配置参数
            bit = config.get('injection_bit')
            location = config.get('injection_location')
            layers = config.get('injection_layers')
            seed = config.get('seed')
            idx = config.get('injection_idx')
            
            # 转换layers为字符串以便分组
            layers_str = str(layers) if layers is not None else 'unknown'
            idx_str = str(idx) if idx is not None else 'unknown'
            
            # 按参数分组
            if bit is not None:
                by_bit[bit].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff,
                    'exp_id': exp_id
                })
            
            if location is not None:
                by_location[location].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff,
                    'exp_id': exp_id
                })
            
            if layers is not None:
                by_layer[layers_str].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff,
                    'exp_id': exp_id
                })
            
            if seed is not None:
                by_seed[seed].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff,
                    'exp_id': exp_id
                })
            
            if idx is not None:
                by_idx[idx_str].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff,
                    'exp_id': exp_id
                })
        
        # 分析按比特位的结果
        print("\n1. Analysis by Bit Position:")
        print("-"*80)
        print(f"{'Bit':<6} {'Count':<8} {'Avg Detection Rate':<25} {'Avg Loss Diff':<20}")
        print("-"*80)
        
        if by_bit:
            for bit in sorted(by_bit.keys()):
                data = by_bit[bit]
                # 过滤NaN值
                detection_rates = [d['detection_rate'] for d in data if not np.isnan(d['detection_rate'])]
                loss_diffs = [d['loss_diff'] for d in data if not np.isnan(d['loss_diff'])]
                
                avg_detection = np.mean(detection_rates) if detection_rates else 0
                avg_loss = np.mean(loss_diffs) if loss_diffs else 0
                print(f"{bit:<6} {len(data):<8} {avg_detection*100:>6.2f}%" + " "*18 + f"{avg_loss:>10.6f}")
        else:
            print("No data available")
        
        # 分析按注入位置的结果
        print("\n2. Analysis by Injection Location:")
        print("-"*80)
        print(f"{'Location':<12} {'Count':<8} {'Avg Detection Rate':<25} {'Avg Loss Diff':<20}")
        print("-"*80)
        
        if by_location:
            for location in sorted(by_location.keys()):
                data = by_location[location]
                # 过滤NaN值
                detection_rates = [d['detection_rate'] for d in data if not np.isnan(d['detection_rate'])]
                loss_diffs = [d['loss_diff'] for d in data if not np.isnan(d['loss_diff'])]
                
                avg_detection = np.mean(detection_rates) if detection_rates else 0
                avg_loss = np.mean(loss_diffs) if loss_diffs else 0
                print(f"{location:<12} {len(data):<8} {avg_detection*100:>6.2f}%" + " "*18 + f"{avg_loss:>10.6f}")
        else:
            print("No data available")
        
        # 分析按层的结果
        print("\n3. Analysis by Injection Layers:")
        print("-"*80)
        print(f"{'Layers':<20} {'Count':<8} {'Avg Detection Rate':<25} {'Avg Loss Diff':<20}")
        print("-"*80)
        
        if by_layer:
            for layers in sorted(by_layer.keys()):
                data = by_layer[layers]
                # 过滤NaN值
                detection_rates = [d['detection_rate'] for d in data if not np.isnan(d['detection_rate'])]
                loss_diffs = [d['loss_diff'] for d in data if not np.isnan(d['loss_diff'])]
                
                avg_detection = np.mean(detection_rates) if detection_rates else 0
                avg_loss = np.mean(loss_diffs) if loss_diffs else 0
                print(f"{layers:<20} {len(data):<8} {avg_detection*100:>6.2f}%" + " "*18 + f"{avg_loss:>10.6f}")
        else:
            print("No data available")
        
        # 分析按种子的结果
        print("\n4. Analysis by Random Seed:")
        print("-"*80)
        print(f"{'Seed':<8} {'Count':<8} {'Avg Detection Rate':<25} {'Avg Loss Diff':<20}")
        print("-"*80)
        
        if by_seed:
            for seed in sorted(by_seed.keys()):
                data = by_seed[seed]
                # 过滤NaN值
                detection_rates = [d['detection_rate'] for d in data if not np.isnan(d['detection_rate'])]
                loss_diffs = [d['loss_diff'] for d in data if not np.isnan(d['loss_diff'])]
                
                avg_detection = np.mean(detection_rates) if detection_rates else 0
                avg_loss = np.mean(loss_diffs) if loss_diffs else 0
                print(f"{seed:<8} {len(data):<8} {avg_detection*100:>6.2f}%" + " "*18 + f"{avg_loss:>10.6f}")
        else:
            print("No data available")
        
        # 分析按注入位置索引的结果
        print("\n5. Analysis by Injection Index:")
        print("-"*80)
        print(f"{'Index':<30} {'Count':<8} {'Avg Detection Rate':<25} {'Avg Loss Diff':<20}")
        print("-"*80)
        
        if by_idx:
            # 只显示前10个
            for idx_str in sorted(by_idx.keys())[:10]:
                data = by_idx[idx_str]
                # 过滤NaN值
                detection_rates = [d['detection_rate'] for d in data if not np.isnan(d['detection_rate'])]
                loss_diffs = [d['loss_diff'] for d in data if not np.isnan(d['loss_diff'])]
                
                avg_detection = np.mean(detection_rates) if detection_rates else 0
                avg_loss = np.mean(loss_diffs) if loss_diffs else 0
                print(f"{idx_str:<30} {len(data):<8} {avg_detection*100:>6.2f}%" + " "*18 + f"{avg_loss:>10.6f}")
            
            if len(by_idx) > 10:
                print(f"... and {len(by_idx) - 10} more")
        else:
            print("No data available")
        
        # 如果有violations数据，额外显示
        if violations_data and 'violations' in violations_data:
            print("\n6. Violations Summary:")
            print("-"*80)
            print(f"Total violation cases: {len(violations_data['violations'])}")
            
            # 统计最常见的违反配置
            violation_configs = defaultdict(int)
            for v in violations_data['violations']:
                config = v.get('config', {})
                bit = config.get('injection_bit')
                location = config.get('injection_location')
                if bit is not None and location is not None:
                    violation_configs[(bit, location)] += 1
            
            if violation_configs:
                print("\nTop 5 violation configurations (bit, location):")
                for (bit, loc), count in sorted(violation_configs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  Bit {bit}, Location {loc}: {count} violations")
        
        # NaN报告
        print(f"\n7. NaN Statistics:")
        print("-"*80)
        print(f"Total experiments with NaN: {total_nan_count}")
        
        # 详细报告：只针对violations中的NaN情况
        if nan_in_violations:
            print(f"\nDetailed NaN Report (violations with NaN): {len(nan_in_violations)} cases")
            print("(NaN in detected violations often indicates bit 30/31 affecting floating-point values)\n")
            
            # 按bit分组统计
            nan_viol_by_bit = defaultdict(list)
            for nan_exp in nan_in_violations:
                bit = nan_exp['config'].get('injection_bit') or nan_exp['config'].get('bit')
                if bit is not None:
                    nan_viol_by_bit[bit].append(nan_exp)
            
            for bit in sorted(nan_viol_by_bit.keys()):
                exps = nan_viol_by_bit[bit]
                print(f"  Bit {bit}: {len(exps)} violation(s) with NaN")
                
                # 显示前3个详细信息
                for exp in exps[:3]:
                    config = exp['config']
                    location = config.get('injection_location') or config.get('location', 'unknown')
                    layers = config.get('injection_layers') or config.get('layers', 'unknown')
                    print(f"    - Location: {location}, Layers: {layers}")
                
                if len(exps) > 3:
                    print(f"    ... and {len(exps) - 3} more")
            
            print(f"\n  Note: These {len(nan_in_violations)} cases had both violations AND NaN values.")
            print(f"        Total NaN cases (including non-violations): {total_nan_count}")
        else:
            print("  No violations with NaN detected.")
            if total_nan_count > 0:
                print(f"  (All {total_nan_count} NaN cases occurred in experiments without violations)")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'by_bit': by_bit,
            'by_location': by_location,
            'by_layer': by_layer,
            'by_seed': by_seed,
            'by_idx': by_idx,
            'violations': violations_data,
            'total_nan_count': total_nan_count,
            'nan_in_violations': nan_in_violations
        }
    
    def find_interesting_cases(self, criterion: str = 'max_violation'):
        """找出有趣的案例"""
        print("="*80)
        print(f"Finding Interesting Cases (criterion: {criterion})")
        print("="*80)
        
        all_cases = []
        
        for exp_id, exp in self.experiments.items():
            if not exp['results']:
                continue
            
            for result in exp['results']:
                case = {
                    'exp_id': exp_id,
                    'run_id': result.get('run_id'),
                    'violation_detected': result.get('violation_detected', False),
                    'num_violations': result.get('num_violations', 0),
                    'loss_diff': result.get('loss_diff', 0)
                }
                
                # 添加epsilon信息
                if 'extra_data' in result and 'epsilon_analysis' in result['extra_data']:
                    max_eps_diff = 0
                    for layer_analysis in result['extra_data']['epsilon_analysis']:
                        analysis = layer_analysis['analysis']
                        max_eps_diff = max(max_eps_diff, abs(analysis['max_abs_diff']))
                    case['max_epsilon_diff'] = max_eps_diff
                else:
                    case['max_epsilon_diff'] = 0
                
                all_cases.append(case)
        
        # 根据不同标准排序
        if criterion == 'max_violation':
            sorted_cases = sorted(all_cases, key=lambda x: x['num_violations'], reverse=True)
        elif criterion == 'max_loss_diff':
            sorted_cases = sorted(all_cases, key=lambda x: abs(x['loss_diff']), reverse=True)
        elif criterion == 'max_epsilon_diff':
            sorted_cases = sorted(all_cases, key=lambda x: x['max_epsilon_diff'], reverse=True)
        else:
            sorted_cases = all_cases
        
        # 显示top 10
        print(f"\nTop 10 cases by {criterion}:")
        print("-"*80)
        print(f"{'Exp ID':<40} {'Run':<6} {'Violations':<12} {'Loss Diff':<15} {'Max Eps Diff':<15}")
        print("-"*80)
        
        for case in sorted_cases[:10]:
            print(f"{case['exp_id']:<40} {case['run_id']:<6} {case['num_violations']:<12} "
                  f"{case['loss_diff']:>10.6f}    {case['max_epsilon_diff']:>10.6f}")
        
        print("\n" + "="*80 + "\n")
        
        return sorted_cases[:10]
    
    def analyze_performance_single(self, exp_id: str):
        """分析单个实验的性能数据 - 重点对比注入和检测的影响"""
        if exp_id not in self.experiments:
            self.load_experiment(exp_id)
        
        if exp_id not in self.experiments:
            print(f"Experiment not found: {exp_id}")
            return None
        
        exp = self.experiments[exp_id]
        results = exp.get('results', [])
        
        if not results:
            print(f"No results found for experiment: {exp_id}")
            return None
        
        print("="*80)
        print(f"Performance Impact Analysis: {exp_id}")
        print("="*80)
        
        # 收集性能数据
        time_data = defaultdict(list)
        memory_data = defaultdict(list)
        
        for result in results:
            if not isinstance(result, dict):
                continue
            
            perf = result.get('extra_data', {}).get('performance', {})
            
            # 时间数据
            if 'time' in perf:
                for key, value in perf['time'].items():
                    if value is not None and not np.isnan(value):
                        time_data[key].append(value)
            
            # 内存数据
            if 'memory' in perf:
                for key, value in perf['memory'].items():
                    if value is not None and not np.isnan(value):
                        memory_data[key].append(value)
        
        if not time_data and not memory_data:
            print("No performance data available")
            return None
        
        # === 1. 错误注入的影响 ===
        print("\n1. Impact of Fault Injection:")
        print("-"*80)
        
        if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
            baseline_time = np.mean(time_data['baseline_forward_time'])
            injection_time = np.mean(time_data['injection_forward_time'])
            time_overhead = injection_time - baseline_time
            time_overhead_pct = (time_overhead / baseline_time * 100) if baseline_time > 0 else 0
            
            print(f"  Time:")
            print(f"    Baseline forward:    {baseline_time:>10.4f} s")
            print(f"    Injection forward:   {injection_time:>10.4f} s")
            print(f"    Overhead:            {time_overhead:>10.4f} s ({time_overhead_pct:>+6.2f}%)")
        else:
            print("  Time data not available")
        
        if 'baseline_memory_mb' in memory_data and 'injection_memory_mb' in memory_data:
            baseline_mem = np.mean(memory_data['baseline_memory_mb'])
            injection_mem = np.mean(memory_data['injection_memory_mb'])
            mem_overhead = injection_mem - baseline_mem
            mem_overhead_pct = (mem_overhead / baseline_mem * 100) if baseline_mem > 0 else 0
            
            print(f"  Memory:")
            print(f"    Baseline memory:     {baseline_mem:>10.2f} MB")
            print(f"    Injection memory:    {injection_mem:>10.2f} MB")
            print(f"    Overhead:            {mem_overhead:>10.2f} MB ({mem_overhead_pct:>+6.2f}%)")
        else:
            print("  Memory data not available")
        
        # === 2. 边界检测的开销 ===
        print("\n2. Overhead of Bounds Detection:")
        print("-"*80)
        
        if 'total_time' in time_data:
            total_time = np.mean(time_data['total_time'])
            
            # 计算核心计算时间（baseline + injection）
            core_time = 0
            if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
                core_time = np.mean(time_data['baseline_forward_time']) + np.mean(time_data['injection_forward_time'])
            
            # 边界计算时间
            bounds_time = np.mean(time_data['bounds_computation_time']) if 'bounds_computation_time' in time_data else 0
            
            # 违反检测时间
            violation_time = np.mean(time_data['violation_detection_time']) if 'violation_detection_time' in time_data else 0
            
            # 总检测开销
            detection_overhead = bounds_time + violation_time
            detection_overhead_pct = (detection_overhead / total_time * 100) if total_time > 0 else 0
            
            print(f"  Time Breakdown:")
            print(f"    Core computation:    {core_time:>10.4f} s ({core_time/total_time*100:>5.1f}%)")
            print(f"    Bounds computation:  {bounds_time:>10.4f} s ({bounds_time/total_time*100:>5.1f}%)")
            print(f"    Violation detection: {violation_time:>10.4f} s ({violation_time/total_time*100:>5.1f}%)")
            print(f"    Total detection:     {detection_overhead:>10.4f} s ({detection_overhead_pct:>5.1f}%)")
            print(f"    Total time:          {total_time:>10.4f} s")
            
            # 如果有核心计算时间，计算相对开销
            if core_time > 0:
                relative_overhead = (detection_overhead / core_time * 100)
                print(f"  Detection overhead relative to core computation: {relative_overhead:.1f}%")
        else:
            print("  Total time data not available")
        
        # 内存开销（如果有peak数据）
        if 'peak_memory_allocated_mb' in memory_data:
            peak_mem = np.mean(memory_data['peak_memory_allocated_mb'])
            baseline_mem = np.mean(memory_data['baseline_memory_mb']) if 'baseline_memory_mb' in memory_data else 0
            
            if baseline_mem > 0:
                mem_overhead_total = peak_mem - baseline_mem
                mem_overhead_pct = (mem_overhead_total / baseline_mem * 100)
                
                print(f"  Memory:")
                print(f"    Baseline memory:     {baseline_mem:>10.2f} MB")
                print(f"    Peak memory:         {peak_mem:>10.2f} MB")
                print(f"    Total overhead:      {mem_overhead_total:>10.2f} MB ({mem_overhead_pct:>+6.2f}%)")
        
        # === 3. 性能摘要 ===
        print("\n3. Performance Summary:")
        print("-"*80)
        
        if 'total_time' in time_data:
            total_time = np.mean(time_data['total_time'])
            std_time = np.std(time_data['total_time'])
            print(f"  Total execution time: {total_time:.4f} ± {std_time:.4f} s")
        
        if 'peak_memory_allocated_mb' in memory_data:
            peak_mem = np.mean(memory_data['peak_memory_allocated_mb'])
            std_mem = np.std(memory_data['peak_memory_allocated_mb'])
            print(f"  Peak memory usage:    {peak_mem:.2f} ± {std_mem:.2f} MB")
        
        # 统计样本数
        num_samples = len(results)
        print(f"  Number of samples:    {num_samples}")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'time_data': dict(time_data),
            'memory_data': dict(memory_data),
            'num_samples': num_samples
        }
    
    def analyze_performance_sweep(self, sweep_dir: str):
        """分析参数扫描中的性能影响趋势"""
        sweep_path = self.results_dir / sweep_dir
        
        if not sweep_path.exists():
            print(f"Sweep directory not found: {sweep_path}")
            return None
        
        print("="*80)
        print(f"Performance Impact Analysis for Sweep: {sweep_dir}")
        print("="*80)
        
        # 收集所有实验
        sweep_experiments = []
        for exp_dir in sweep_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'violations':
                exp_id = exp_dir.name
                if exp_id not in self.experiments:
                    self.load_experiment(exp_id, parent_dir=sweep_path)
                if exp_id in self.experiments:
                    sweep_experiments.append(exp_id)
        
        print(f"\nFound {len(sweep_experiments)} configurations")
        
        # 按参数分组收集性能影响数据
        impact_by_bit = defaultdict(lambda: defaultdict(list))
        impact_by_location = defaultdict(lambda: defaultdict(list))
        impact_by_layer = defaultdict(lambda: defaultdict(list))
        
        for exp_id in sweep_experiments:
            exp = self.experiments[exp_id]
            config = exp.get('config')
            if not config:
                config = self.parse_config_from_exp_id(exp_id)
            
            if not config:
                continue
            
            results = exp.get('results', [])
            
            # 收集该实验的性能数据
            exp_time_data = defaultdict(list)
            exp_memory_data = defaultdict(list)
            
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                perf = result.get('extra_data', {}).get('performance', {})
                
                if 'time' in perf:
                    for key, value in perf['time'].items():
                        if value is not None and not np.isnan(value):
                            exp_time_data[key].append(value)
                
                if 'memory' in perf:
                    for key, value in perf['memory'].items():
                        if value is not None and not np.isnan(value):
                            exp_memory_data[key].append(value)
            
            # 计算注入影响和检测开销
            injection_time_overhead = 0
            injection_mem_overhead = 0
            detection_time_overhead = 0
            detection_time_pct = 0
            
            if 'baseline_forward_time' in exp_time_data and 'injection_forward_time' in exp_time_data:
                baseline_t = np.mean(exp_time_data['baseline_forward_time'])
                injection_t = np.mean(exp_time_data['injection_forward_time'])
                injection_time_overhead = ((injection_t - baseline_t) / baseline_t * 100) if baseline_t > 0 else 0
            
            if 'baseline_memory_mb' in exp_memory_data and 'injection_memory_mb' in exp_memory_data:
                baseline_m = np.mean(exp_memory_data['baseline_memory_mb'])
                injection_m = np.mean(exp_memory_data['injection_memory_mb'])
                injection_mem_overhead = ((injection_m - baseline_m) / baseline_m * 100) if baseline_m > 0 else 0
            
            if 'total_time' in exp_time_data:
                total_t = np.mean(exp_time_data['total_time'])
                bounds_t = np.mean(exp_time_data['bounds_computation_time']) if 'bounds_computation_time' in exp_time_data else 0
                violation_t = np.mean(exp_time_data['violation_detection_time']) if 'violation_detection_time' in exp_time_data else 0
                detection_time_overhead = bounds_t + violation_t
                detection_time_pct = (detection_time_overhead / total_t * 100) if total_t > 0 else 0
            
            # 按参数分组
            bit = config.get('injection_bit') or config.get('bit')
            location = config.get('injection_location') or config.get('location')
            layers = str(config.get('injection_layers') or config.get('layers', ''))
            
            if bit is not None:
                impact_by_bit[bit]['injection_time_overhead'].append(injection_time_overhead)
                impact_by_bit[bit]['injection_mem_overhead'].append(injection_mem_overhead)
                impact_by_bit[bit]['detection_time_overhead'].append(detection_time_overhead)
                impact_by_bit[bit]['detection_time_pct'].append(detection_time_pct)
            
            if location:
                impact_by_location[location]['injection_time_overhead'].append(injection_time_overhead)
                impact_by_location[location]['injection_mem_overhead'].append(injection_mem_overhead)
                impact_by_location[location]['detection_time_overhead'].append(detection_time_overhead)
                impact_by_location[location]['detection_time_pct'].append(detection_time_pct)
            
            if layers:
                impact_by_layer[layers]['injection_time_overhead'].append(injection_time_overhead)
                impact_by_layer[layers]['injection_mem_overhead'].append(injection_mem_overhead)
                impact_by_layer[layers]['detection_time_overhead'].append(detection_time_overhead)
                impact_by_layer[layers]['detection_time_pct'].append(detection_time_pct)
        
        # 分析并显示结果
        print("\n1. Fault Injection Impact by Bit Position:")
        print("-"*80)
        if impact_by_bit:
            print(f"{'Bit':<6} {'Count':<8} {'Time Overhead':<20} {'Memory Overhead':<20}")
            print("-"*80)
            for bit in sorted(impact_by_bit.keys()):
                data = impact_by_bit[bit]
                count = len(data['injection_time_overhead'])
                avg_time_oh = np.mean(data['injection_time_overhead']) if data['injection_time_overhead'] else 0
                avg_mem_oh = np.mean(data['injection_mem_overhead']) if data['injection_mem_overhead'] else 0
                print(f"{bit:<6} {count:<8} {avg_time_oh:>+8.2f}%{'':<11} {avg_mem_oh:>+8.2f}%")
        else:
            print("No data available")
        
        print("\n2. Fault Injection Impact by Injection Location:")
        print("-"*80)
        if impact_by_location:
            print(f"{'Location':<12} {'Count':<8} {'Time Overhead':<20} {'Memory Overhead':<20}")
            print("-"*80)
            for location in sorted(impact_by_location.keys()):
                data = impact_by_location[location]
                count = len(data['injection_time_overhead'])
                avg_time_oh = np.mean(data['injection_time_overhead']) if data['injection_time_overhead'] else 0
                avg_mem_oh = np.mean(data['injection_mem_overhead']) if data['injection_mem_overhead'] else 0
                print(f"{location:<12} {count:<8} {avg_time_oh:>+8.2f}%{'':<11} {avg_mem_oh:>+8.2f}%")
        else:
            print("No data available")
        
        print("\n3. Bounds Detection Overhead by Bit Position:")
        print("-"*80)
        if impact_by_bit:
            print(f"{'Bit':<6} {'Count':<8} {'Detection Time (s)':<20} {'Detection Time %':<20}")
            print("-"*80)
            for bit in sorted(impact_by_bit.keys()):
                data = impact_by_bit[bit]
                count = len(data['detection_time_overhead'])
                avg_det_time = np.mean(data['detection_time_overhead']) if data['detection_time_overhead'] else 0
                avg_det_pct = np.mean(data['detection_time_pct']) if data['detection_time_pct'] else 0
                print(f"{bit:<6} {count:<8} {avg_det_time:<20.4f} {avg_det_pct:>6.2f}%")
        else:
            print("No data available")
        
        print("\n4. Bounds Detection Overhead by Injection Location:")
        print("-"*80)
        if impact_by_location:
            print(f"{'Location':<12} {'Count':<8} {'Detection Time (s)':<20} {'Detection Time %':<20}")
            print("-"*80)
            for location in sorted(impact_by_location.keys()):
                data = impact_by_location[location]
                count = len(data['detection_time_overhead'])
                avg_det_time = np.mean(data['detection_time_overhead']) if data['detection_time_overhead'] else 0
                avg_det_pct = np.mean(data['detection_time_pct']) if data['detection_time_pct'] else 0
                print(f"{location:<12} {count:<8} {avg_det_time:<20.4f} {avg_det_pct:>6.2f}%")
        else:
            print("No data available")
        
        print("\n5. Overall Performance Impact Summary:")
        print("-"*80)
        
        # 计算总体统计
        all_injection_time_oh = []
        all_injection_mem_oh = []
        all_detection_time = []
        all_detection_pct = []
        
        for data in impact_by_bit.values():
            all_injection_time_oh.extend(data['injection_time_overhead'])
            all_injection_mem_oh.extend(data['injection_mem_overhead'])
            all_detection_time.extend(data['detection_time_overhead'])
            all_detection_pct.extend(data['detection_time_pct'])
        
        if all_injection_time_oh:
            print(f"  Fault injection time overhead:   {np.mean(all_injection_time_oh):>+7.2f}% ± {np.std(all_injection_time_oh):>6.2f}%")
        if all_injection_mem_oh:
            print(f"  Fault injection memory overhead:  {np.mean(all_injection_mem_oh):>+7.2f}% ± {np.std(all_injection_mem_oh):>6.2f}%")
        if all_detection_time:
            print(f"  Bounds detection time:            {np.mean(all_detection_time):>7.4f} ± {np.std(all_detection_time):>6.4f} s")
        if all_detection_pct:
            print(f"  Bounds detection percentage:      {np.mean(all_detection_pct):>7.2f}% ± {np.std(all_detection_pct):>6.2f}%")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'by_bit': dict(impact_by_bit),
            'by_location': dict(impact_by_location),
            'by_layer': dict(impact_by_layer)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('results_dir', type=str, help='Results directory path')
    parser.add_argument('--exp-id', type=str, help='Specific experiment ID to analyze')
    parser.add_argument('--sweep', type=str, help='Parameter sweep directory to analyze')
    parser.add_argument('--compare', nargs='+', help='List of experiment IDs to compare')
    parser.add_argument('--interesting', type=str, 
                       choices=['max_violation', 'max_loss_diff', 'max_epsilon_diff'],
                       help='Find interesting cases')
    parser.add_argument('--perf', type=str, metavar='EXP_ID',
                       help='Analyze performance for a single experiment')
    parser.add_argument('--perf-sweep', type=str, metavar='SWEEP_DIR',
                       help='Analyze performance for a parameter sweep')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if args.exp_id:
        # 分析单个实验
        analyzer.analyze_single_experiment(args.exp_id)
    
    elif args.sweep:
        # 分析参数扫描
        analyzer.analyze_parameter_sweep(args.sweep)
    
    elif args.compare:
        # 对比多个实验
        analyzer.compare_experiments(args.compare)
    
    elif args.interesting:
        # 找有趣的案例
        analyzer.load_all_experiments()
        analyzer.find_interesting_cases(args.interesting)
    
    elif args.perf:
        # 分析单个实验的性能
        analyzer.analyze_performance_single(args.perf)
    
    elif args.perf_sweep:
        # 分析参数扫描的性能
        analyzer.analyze_performance_sweep(args.perf_sweep)
    
    else:
        # 加载并列出所有实验
        analyzer.load_all_experiments()
        print("\nAvailable experiments:")
        for exp_id in sorted(analyzer.experiments.keys()):
            print(f"  - {exp_id}")
        print("\nUse --exp-id to analyze a specific experiment")
        print("Use --sweep to analyze a parameter sweep")
        print("Use --compare to compare multiple experiments")
        print("Use --interesting to find interesting cases")
        print("Use --perf to analyze performance for a single experiment")
        print("Use --perf-sweep to analyze performance for a parameter sweep")


if __name__ == "__main__":
    main()