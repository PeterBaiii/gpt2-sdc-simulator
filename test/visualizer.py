"""
结果可视化工具 - 可视化实验JSON结果
"""

import json
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import seaborn as sns


# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 结果目录路径
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        
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
    
    def visualize_single_experiment(self, exp_id: str, save_path: Optional[str] = None):
        """可视化单个实验"""
        if exp_id not in self.experiments:
            self.load_experiment(exp_id)
        
        exp = self.experiments[exp_id]
        results = exp['results']
        
        if not results:
            print(f"No results found for {exp_id}")
            return
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Experiment Results: {exp_id}', fontsize=16, fontweight='bold')
        
        # 1. Loss对比 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        baseline_losses = [r['baseline_loss'] for r in results if r.get('baseline_loss') is not None]
        injected_losses = [r['injected_loss'] for r in results if r.get('injected_loss') is not None]
        
        x = np.arange(len(baseline_losses))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_losses, width, label='Baseline', alpha=0.8)
        ax1.bar(x + width/2, injected_losses, width, label='Injected', alpha=0.8)
        
        ax1.set_xlabel('Run ID')
        ax1.set_ylabel('Loss')
        ax1.set_title('Baseline vs Injected Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss差异 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        loss_diffs = [r['loss_diff'] for r in results if r.get('loss_diff') is not None]
        
        ax2.plot(loss_diffs, 'o-', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.fill_between(range(len(loss_diffs)), 0, loss_diffs, alpha=0.3)
        
        ax2.set_xlabel('Run ID')
        ax2.set_ylabel('Loss Difference')
        ax2.set_title('Loss Difference (Injected - Baseline)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 违反检测统计 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        violations = [r.get('violation_detected', False) for r in results]
        detected_count = sum(violations)
        not_detected_count = len(violations) - detected_count
        
        colors = ['#ff6b6b', '#51cf66']
        sizes = [detected_count, not_detected_count]
        labels = [f'Violation Detected\n({detected_count})', 
                 f'No Violation\n({not_detected_count})']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Violation Detection Rate')
        
        # 4. 违反数量分布 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        num_violations = [r.get('num_violations', 0) for r in results]
        
        ax4.bar(range(len(num_violations)), num_violations, alpha=0.8)
        ax4.set_xlabel('Run ID')
        ax4.set_ylabel('Number of Violations')
        ax4.set_title('Violation Count per Run')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Epsilon差异分布 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        all_epsilon_diffs = []
        
        for r in results:
            if 'extra_data' in r and 'epsilon_analysis' in r['extra_data']:
                for layer_analysis in r['extra_data']['epsilon_analysis']:
                    analysis = layer_analysis['analysis']
                    all_epsilon_diffs.append(analysis['max_abs_diff'])
        
        if all_epsilon_diffs:
            ax5.hist(all_epsilon_diffs, bins=30, alpha=0.7, edgecolor='black')
            ax5.axvline(float(np.mean(all_epsilon_diffs)), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(all_epsilon_diffs):.4f}')
            ax5.set_xlabel('Max |Epsilon Diff|')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Distribution of Max Epsilon Differences')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 性能时间分析 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        perf_data = []
        
        for r in results:
            if 'extra_data' in r and 'performance' in r['extra_data']:
                perf = r['extra_data']['performance']
                perf_data.append({
                    'baseline': perf['time']['baseline_forward_time'],
                    'injection': perf['time']['injection_forward_time'],
                    'bounds': perf['time']['bounds_computation_time'],
                    'detection': perf['time']['violation_detection_time']
                })
        
        if perf_data:
            categories = ['Baseline\nForward', 'Injection\nForward', 
                         'Bounds\nComputation', 'Violation\nDetection']
            
            baseline_times = [p['baseline'] for p in perf_data]
            injection_times = [p['injection'] for p in perf_data]
            bounds_times = [p['bounds'] for p in perf_data]
            detection_times = [p['detection'] for p in perf_data]
            
            times = [np.mean(baseline_times), np.mean(injection_times),
                    np.mean(bounds_times), np.mean(detection_times)]
            
            bars = ax6.bar(categories, times, alpha=0.8)
            bars[0].set_color('#3498db')
            bars[1].set_color('#e74c3c')
            bars[2].set_color('#f39c12')
            bars[3].set_color('#9b59b6')
            
            ax6.set_ylabel('Time (seconds)')
            ax6.set_title('Average Time Breakdown')
            ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. 每层的epsilon分析 (左下，跨2列)
        ax7 = fig.add_subplot(gs[2, :2])
        
        layer_epsilon_data = defaultdict(list)
        
        for r in results:
            if 'extra_data' in r and 'epsilon_analysis' in r['extra_data']:
                for layer_analysis in r['extra_data']['epsilon_analysis']:
                    layer_idx = layer_analysis['layer_idx']
                    analysis = layer_analysis['analysis']
                    layer_epsilon_data[layer_idx].append({
                        'mean': analysis['mean_diff'],
                        'max': analysis['max_abs_diff']
                    })
        
        if layer_epsilon_data:
            layers = sorted(layer_epsilon_data.keys())
            mean_diffs = [np.mean([d['mean'] for d in layer_epsilon_data[l]]) for l in layers]
            max_diffs = [np.mean([d['max'] for d in layer_epsilon_data[l]]) for l in layers]
            
            x_pos = np.arange(len(layers))
            
            ax7.plot(x_pos, mean_diffs, 'o-', label='Mean Epsilon Diff', linewidth=2, markersize=6)
            ax7.plot(x_pos, max_diffs, 's-', label='Max |Epsilon Diff|', linewidth=2, markersize=6)
            
            ax7.set_xlabel('Layer Index')
            ax7.set_ylabel('Epsilon Difference')
            ax7.set_title('Epsilon Difference by Layer')
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(layers)
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. 内存使用 (右下)
        ax8 = fig.add_subplot(gs[2, 2])
        
        memory_data = []
        for r in results:
            if 'extra_data' in r and 'performance' in r['extra_data']:
                mem = r['extra_data']['performance']['memory']
                memory_data.append({
                    'peak': mem['peak_memory_allocated_mb'],
                    'baseline': mem['baseline_memory_mb'],
                    'injection': mem['injection_memory_mb']
                })
        
        if memory_data:
            categories = ['Peak', 'Baseline', 'Injection']
            values = [
                np.mean([m['peak'] for m in memory_data]),
                np.mean([m['baseline'] for m in memory_data]),
                np.mean([m['injection'] for m in memory_data])
            ]
            
            bars = ax8.bar(categories, values, alpha=0.8)
            bars[0].set_color('#e74c3c')
            bars[1].set_color('#3498db')
            bars[2].set_color('#f39c12')
            
            ax8.set_ylabel('Memory (MB)')
            ax8.set_title('Average Memory Usage')
            ax8.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            save_path = self.experiments[exp_id]['exp_dir'] / "visualization.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def visualize_parameter_sweep(self, sweep_dir: str, save_path: Optional[str] = None):
        """可视化参数扫描结果"""
        sweep_path = self.results_dir / sweep_dir
        
        if not sweep_path.exists():
            print(f"Sweep directory not found: {sweep_path}")
            return
        
        # 加载所有实验
        sweep_experiments = []
        for exp_dir in sweep_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'violations':
                exp_id = exp_dir.name
                if exp_id not in self.experiments:
                    self.load_experiment(exp_id, parent_dir=sweep_path)
                if exp_id in self.experiments:
                    sweep_experiments.append(exp_id)
        
        print(f"Loaded {len(sweep_experiments)} sweep configurations")
        
        # 收集数据
        by_bit = defaultdict(list)
        by_location = defaultdict(list)
        by_layer = defaultdict(list)
        
        # 记录NaN情况
        nan_count = 0
        
        for exp_id in sweep_experiments:
            exp = self.experiments[exp_id]
            
            # 优先使用config，如果没有则尝试解析exp_id
            config = exp.get('config')
            if not config:
                config = self.parse_config_from_exp_id(exp_id)
            
            if not config:
                continue
            
            summary = exp.get('summary')
            
            detection_rate = 0
            mean_loss_diff = 0
            
            if summary:
                if 'violation_detection' in summary:
                    detection_rate = summary['violation_detection'].get('detection_rate', 0)
                if 'loss_diff' in summary:
                    mean_loss_diff = summary['loss_diff'].get('mean', 0)
                    if np.isnan(mean_loss_diff):
                        nan_count += 1
                        continue  # 跳过NaN数据点以避免影响可视化
            else:
                # 如果没有summary，尝试从results计算
                results = exp.get('results', [])
                if results:
                    violations = [r.get('violation_detected', False) for r in results if isinstance(r, dict)]
                    if violations:
                        detection_rate = sum(violations) / len(violations)
                    
                    # 过滤NaN值
                    loss_diffs = []
                    for r in results:
                        if isinstance(r, dict) and r.get('loss_diff') is not None:
                            raw = r.get('loss_diff')
                            
                            if raw is None:
                                continue
                            try:
                                diff = float(raw)
                            except Exception:
                                continue
                            
                            if not np.isnan(diff):
                                loss_diffs.append(diff)
                            else:
                                nan_count += 1
                    
                    if loss_diffs:
                        mean_loss_diff = np.mean(loss_diffs)
                    else:
                        continue  # 全是NaN，跳过
            
            # 检查detection_rate是否为NaN
            if np.isnan(detection_rate):
                nan_count += 1
                continue
            
            bit = config.get('injection_bit')
            location = config.get('injection_location')
            layers = str(config.get('injection_layers', ''))
            
            if bit is not None:
                by_bit[bit].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff
                })
            
            if location is not None:
                by_location[location].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff
                })
            
            if layers:
                by_layer[layers].append({
                    'detection_rate': detection_rate,
                    'loss_diff': mean_loss_diff
                })
        
        if not by_bit and not by_location:
            print("No valid data found for visualization")
            return
        
        # 如果有NaN，打印简要信息
        if nan_count > 0:
            print(f"\n⚠ {nan_count} experiments with NaN values were excluded from visualization")
        
        # 创建图表
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        title = f'Parameter Sweep Results: {sweep_dir}'
        if nan_count > 0:
            title += f' ({nan_count} NaN excluded)'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 按比特位的检测率 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        
        if by_bit:
            bits = sorted(by_bit.keys())
            detection_rates = [np.mean([d['detection_rate'] for d in by_bit[b]]) * 100 for b in bits]
            
            ax1.plot(bits, detection_rates, 'o-', linewidth=2, markersize=6)
            ax1.set_xlabel('Bit Position')
            ax1.set_ylabel('Detection Rate (%)')
            ax1.set_title('Detection Rate by Bit Position')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            
            # 标注符号位
            # if 31 in bits:
            #     ax1.axvline(x=31, color='r', linestyle='--', alpha=0.5)
            #     ax1.text(31, max(detection_rates) * 0.9, 'Sign bit', 
            #             ha='center', fontsize=9, color='r')
        
        # 2. 按比特位的loss差异 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if by_bit:
            bits = sorted(by_bit.keys())
            loss_diffs = [np.mean([abs(d['loss_diff']) for d in by_bit[b]]) for b in bits]
            
            ax2.bar(bits, loss_diffs, alpha=0.7)
            ax2.set_xlabel('Bit Position')
            ax2.set_ylabel('Avg |Loss Diff|')
            ax2.set_title('Average Absolute Loss Difference by Bit')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 按注入位置的对比 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        
        if by_location:
            locations = sorted(by_location.keys())
            detection_rates = [np.mean([d['detection_rate'] for d in by_location[l]]) * 100 
                             for l in locations]
            
            bars = ax3.bar(locations, detection_rates, alpha=0.8)
            
            # 颜色编码
            cmap = plt.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, len(locations)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax3.set_ylabel('Detection Rate (%)')
            ax3.set_title('Detection Rate by Injection Location')
            ax3.set_ylim(0, 105)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 旋转x标签
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. 按层的检测率 (左下)
        ax4 = fig.add_subplot(gs[1, 0])
        
        if by_layer:
            layers = sorted(by_layer.keys())[:10]  # 只显示前10个
            detection_rates = [np.mean([d['detection_rate'] for d in by_layer[l]]) * 100 
                             for l in layers]
            
            ax4.barh(range(len(layers)), detection_rates, alpha=0.8)
            ax4.set_yticks(range(len(layers)))
            ax4.set_yticklabels(layers, fontsize=9)
            ax4.set_xlabel('Detection Rate (%)')
            ax4.set_title('Detection Rate by Layer Configuration (Top 10)')
            ax4.set_xlim(0, 105)
            ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. 热力图：比特位 × 注入位置 (中下)
        ax5 = fig.add_subplot(gs[1, 1])
        
        if by_bit and by_location:
            bits = sorted(by_bit.keys())
            locations = sorted(by_location.keys())
            
            # 重新组织数据为二维数组
            heatmap_data = np.zeros((len(locations), len(bits)))
            count_data = np.zeros((len(locations), len(bits)))
            
            for exp_id in sweep_experiments:
                exp = self.experiments[exp_id]
                config = exp.get('config')
                if not config:
                    config = self.parse_config_from_exp_id(exp_id)
                
                if not config:
                    continue
                
                summary = exp.get('summary')
                
                bit = config.get('injection_bit') or config.get('bit')
                location = config.get('injection_location') or config.get('location')
                
                if bit in bits and location in locations:
                    detection_rate = 0
                    if summary and 'violation_detection' in summary:
                        detection_rate = summary['violation_detection'].get('detection_rate', 0)
                    elif exp.get('results'):
                        results = exp.get('results', [])
                        violations = [r.get('violation_detected', False) for r in results if isinstance(r, dict)]
                        if violations:
                            detection_rate = sum(violations) / len(violations)
                    
                    # 跳过NaN值
                    if np.isnan(detection_rate):
                        continue
                    
                    bit_idx = bits.index(bit)
                    loc_idx = locations.index(location)
                    heatmap_data[loc_idx, bit_idx] += detection_rate
                    count_data[loc_idx, bit_idx] += 1
            
            # 平均化
            mask = count_data > 0
            heatmap_data[mask] /= count_data[mask]
            heatmap_data *= 100  # 转换为百分比
            
            # 将没有数据的地方设为NaN以便在热力图上显示
            heatmap_data[~mask] = np.nan
            
            im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            ax5.set_xticks(np.arange(len(bits)))
            ax5.set_yticks(np.arange(len(locations)))
            ax5.set_xticklabels(bits, fontsize=5)
            ax5.set_yticklabels(locations, fontsize=10)
            
            ax5.set_xlabel('Bit Position')
            ax5.set_ylabel('Injection Location')
            ax5.set_title('Detection Rate Heatmap (%)')
            
            plt.colorbar(im, ax=ax5, label='Detection Rate (%)')
        
        # 6. Loss差异的箱线图 (右下)
        ax6 = fig.add_subplot(gs[1, 2])
        
        if by_location:
            locations = sorted(by_location.keys())
            # 过滤NaN值
            loss_data = [[d['loss_diff'] for d in by_location[l] if not np.isnan(d['loss_diff'])] 
                        for l in locations]
            
            # 只保留有数据的位置
            valid_locations = []
            valid_loss_data = []
            for loc, data in zip(locations, loss_data):
                if data:  # 只保留非空数据
                    valid_locations.append(loc)
                    valid_loss_data.append(data)
            
            if valid_loss_data:
                bp = ax6.boxplot(valid_loss_data, patch_artist=True)
                ax6.set_xticks(range(1, len(valid_locations) + 1))
                ax6.set_xticklabels(valid_locations)
            
                # 美化箱线图
                colors = colormaps['Set3'](np.linspace(0, 1, len(valid_locations)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax6.set_ylabel('Loss Difference')
                ax6.set_title('Loss Difference Distribution by Location')
                ax6.grid(True, alpha=0.3, axis='y')
                
                plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            save_path = str(sweep_path / "sweep_visualization.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def visualize_epsilon_details(self, exp_id: str, run_id: int = 0, 
                                  save_path: Optional[str] = None):
        """详细可视化epsilon分析"""
        if exp_id not in self.experiments:
            self.load_experiment(exp_id)
        
        exp = self.experiments[exp_id]
        results = exp['results']
        
        if run_id >= len(results):
            print(f"Run {run_id} not found (total runs: {len(results)})")
            return
        
        result = results[run_id]
        
        if 'extra_data' not in result or 'epsilon_analysis' not in result['extra_data']:
            print(f"No epsilon analysis found for run {run_id}")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Epsilon Analysis Details: {exp_id}, Run {run_id}', 
                    fontsize=14, fontweight='bold')
        
        epsilon_data = result['extra_data']['epsilon_analysis']
        
        # 1. 每层的统计 (左上)
        ax1 = axes[0, 0]
        
        layers = []
        mean_diffs = []
        std_diffs = []
        max_diffs = []
        
        for layer_analysis in epsilon_data:
            layers.append(layer_analysis['layer_idx'])
            analysis = layer_analysis['analysis']
            mean_diffs.append(analysis['mean_diff'])
            std_diffs.append(analysis['std_diff'])
            max_diffs.append(analysis['max_abs_diff'])
        
        x = np.arange(len(layers))
        
        ax1.errorbar(x, mean_diffs, yerr=std_diffs, fmt='o-', 
                    linewidth=2, markersize=6, capsize=5, label='Mean ± Std')
        ax1.plot(x, max_diffs, 's--', linewidth=2, markersize=6, 
                label='Max |Diff|', alpha=0.7)
        
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Epsilon Difference')
        ax1.set_title('Epsilon Statistics by Layer')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top变化位置 (右上)
        ax2 = axes[0, 1]
        
        # 收集所有top changes
        all_changes = []
        for layer_analysis in epsilon_data:
            layer_idx = layer_analysis['layer_idx']
            for pos, changes, bounds in layer_analysis['analysis']['top_changes'][:5]:
                all_changes.append({
                    'layer': layer_idx,
                    'position': str(pos),
                    'abs_diff': changes['abs_diff'],
                    'baseline': changes['baseline_epsilon'],
                    'injected': changes['injected_epsilon']
                })
        
        # 显示top 10
        all_changes_sorted = sorted(all_changes, key=lambda x: x['abs_diff'], reverse=True)[:10]
        
        labels = [f"L{c['layer']}\n{c['position'][:10]}" for c in all_changes_sorted]
        values = [c['abs_diff'] for c in all_changes_sorted]
        
        ax2.barh(range(len(labels)), values, alpha=0.8)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel('|Epsilon Diff|')
        ax2.set_title('Top 10 Epsilon Changes')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Baseline vs Injected epsilon (左下)
        ax3 = axes[1, 0]
        
        baseline_eps = [c['baseline'] for c in all_changes_sorted]
        injected_eps = [c['injected'] for c in all_changes_sorted]
        
        x_pos = np.arange(len(baseline_eps))
        width = 0.35
        
        ax3.bar(x_pos - width/2, baseline_eps, width, label='Baseline', alpha=0.8)
        ax3.bar(x_pos + width/2, injected_eps, width, label='Injected', alpha=0.8)
        
        ax3.set_xlabel('Top Change Index')
        ax3.set_ylabel('Epsilon Value')
        ax3.set_title('Baseline vs Injected Epsilon (Top 10)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Bounds violation visualization (右下)
        ax4 = axes[1, 1]
        
        # 显示bounds和epsilon的关系
        layer_idx = epsilon_data[0]['layer_idx']  # 使用第一层作为示例
        top_changes = epsilon_data[0]['analysis']['top_changes'][:5]
        
        positions = [f"Pos{i+1}" for i in range(len(top_changes))]
        
        lower_bounds = [bounds['middle'] for _, _, bounds in top_changes]
        upper_bounds = [bounds['upper'] for _, _, bounds in top_changes]
        epsilon_vals = [changes['injected_epsilon'] for _, changes, _ in top_changes]
        
        x_pos = np.arange(len(positions))
        
        # 画bounds范围
        for i, (lower, upper, eps) in enumerate(zip(lower_bounds, upper_bounds, epsilon_vals)):
            ax4.plot([i, i], [lower, upper], 'b-', linewidth=8, alpha=0.3)
            ax4.plot(i, eps, 'ro', markersize=10)
            
            # 标注是否违反
            if eps < lower or eps > upper:
                ax4.plot(i, eps, 'r*', markersize=15, markeredgecolor='black', markeredgewidth=1.5)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(positions)
        ax4.set_ylabel('Value')
        ax4.set_title(f'Bounds vs Epsilon (Layer {layer_idx}, Top 5)')
        ax4.legend(['Bounds Range', 'Epsilon', 'Violation'], loc='best')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            save_path = self.experiments[exp_id]['exp_dir'] / f"epsilon_details_run{run_id}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def visualize_performance_single(self, exp_id: str, save_path: Optional[str] = None):
        """可视化单个实验的性能影响 - 对比注入和检测的开销"""
        if exp_id not in self.experiments:
            self.load_experiment(exp_id)
        
        if exp_id not in self.experiments:
            print(f"Experiment not found: {exp_id}")
            return
        
        exp = self.experiments[exp_id]
        results = exp.get('results', [])
        
        if not results:
            print(f"No results found for experiment: {exp_id}")
            return
        
        # 收集性能数据
        time_data = defaultdict(list)
        memory_data = defaultdict(list)
        
        for result in results:
            if not isinstance(result, dict):
                continue
            
            perf = result.get('extra_data', {}).get('performance', {})
            
            if 'time' in perf:
                for key, value in perf['time'].items():
                    if value is not None and not np.isnan(value):
                        time_data[key].append(value)
            
            if 'memory' in perf:
                for key, value in perf['memory'].items():
                    if value is not None and not np.isnan(value):
                        memory_data[key].append(value)
        
        if not time_data and not memory_data:
            print("No performance data available for visualization")
            return
        
        # 创建图表 (2x2 布局)
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Performance Impact Analysis: {exp_id[:60]}', fontsize=14, fontweight='bold')
        
        # 1. 注入影响对比（时间和内存）- 柱状图
        ax1 = fig.add_subplot(gs[0, 0])
        
        if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
            baseline_time = np.mean(time_data['baseline_forward_time'])
            injection_time = np.mean(time_data['injection_forward_time'])
            
            baseline_mem = np.mean(memory_data['baseline_memory_mb']) if 'baseline_memory_mb' in memory_data else 0
            injection_mem = np.mean(memory_data['injection_memory_mb']) if 'injection_memory_mb' in memory_data else 0
            
            # 双Y轴图
            ax1_twin = ax1.twinx()
            
            x = np.array([0, 1])
            width = 0.35
            
            # 时间柱
            bars1 = ax1.bar(x - width/2, np.array([baseline_time, injection_time]), width, 
                           label='Time', color='steelblue', alpha=0.8)
            
            # 内存柱
            bars2 = ax1_twin.bar(x + width/2, np.array([baseline_mem, injection_mem]), width,
                                label='Memory', color='coral', alpha=0.8)
            
            ax1.set_ylabel('Time (s)', color='steelblue')
            ax1_twin.set_ylabel('Memory (MB)', color='coral')
            ax1.set_xticks(x)
            ax1.set_xticklabels(['Baseline', 'Injection'])
            ax1.set_title('Fault Injection Impact')
            ax1.tick_params(axis='y', labelcolor='steelblue')
            ax1_twin.tick_params(axis='y', labelcolor='coral')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s',
                        ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax1_twin.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.0f}MB',
                             ha='center', va='bottom', fontsize=9)
        
        # 2. 时间分解（堆叠柱状图）
        ax2 = fig.add_subplot(gs[0, 1])
        
        if 'total_time' in time_data:
            components = []
            labels = []
            colors_list = []
            
            # 核心计算
            if 'baseline_forward_time' in time_data:
                components.append(np.mean(time_data['baseline_forward_time']))
                labels.append('Baseline Forward')
                colors_list.append('lightblue')
            
            if 'injection_forward_time' in time_data:
                components.append(np.mean(time_data['injection_forward_time']))
                labels.append('Injection Forward')
                colors_list.append('steelblue')
            
            # 检测开销
            if 'bounds_computation_time' in time_data:
                components.append(np.mean(time_data['bounds_computation_time']))
                labels.append('Bounds Computation')
                colors_list.append('orange')
            
            if 'violation_detection_time' in time_data:
                components.append(np.mean(time_data['violation_detection_time']))
                labels.append('Violation Detection')
                colors_list.append('coral')
            
            if components:
                # 堆叠柱状图
                bottom = 0
                for comp, label, color in zip(components, labels, colors_list):
                    bar = ax2.bar(0, comp, bottom=bottom, label=label, color=color, alpha=0.8, width=0.5)
                    # 添加标签
                    if comp > 0.01:  # 只标注大于0.01的
                        pct = comp / sum(components) * 100
                        ax2.text(0, bottom + comp/2, f'{comp:.3f}s\n({pct:.1f}%)',
                                ha='center', va='center', fontsize=9, fontweight='bold')
                    bottom += comp
                
                ax2.set_ylabel('Time (s)')
                ax2.set_title('Time Breakdown')
                ax2.set_xticks([])
                ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
                ax2.set_ylim(0, sum(components) * 1.1)
        
        # 3. 检测开销占比（饼图）
        ax3 = fig.add_subplot(gs[1, 0])
        
        if 'total_time' in time_data:
            total_time = np.mean(time_data['total_time'])
            
            core_time = 0
            if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
                core_time = np.mean(time_data['baseline_forward_time']) + np.mean(time_data['injection_forward_time'])
            
            bounds_time = np.mean(time_data['bounds_computation_time']) if 'bounds_computation_time' in time_data else 0
            violation_time = np.mean(time_data['violation_detection_time']) if 'violation_detection_time' in time_data else 0
            
            if core_time > 0 and (bounds_time > 0 or violation_time > 0):
                sizes = np.array([core_time, bounds_time, violation_time])
                labels_pie = ['Core Computation', 'Bounds Computation', 'Violation Detection']
                colors_pie = ['lightblue', 'orange', 'coral']
                explode = (0, 0.1, 0.1)
                
                ax3.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie,
                       explode=explode, startangle=90, textprops={'fontsize': 10})
                ax3.set_title('Detection Overhead Proportion')
        
        # 4. 开销百分比对比（柱状图）
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
            baseline_time = np.mean(time_data['baseline_forward_time'])
            injection_time = np.mean(time_data['injection_forward_time'])
            time_overhead_pct = ((injection_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
            
            baseline_mem = np.mean(memory_data['baseline_memory_mb']) if 'baseline_memory_mb' in memory_data else 0
            injection_mem = np.mean(memory_data['injection_memory_mb']) if 'injection_memory_mb' in memory_data else 0
            mem_overhead_pct = ((injection_mem - baseline_mem) / baseline_mem * 100) if baseline_mem > 0 else 0
            
            total_time = np.mean(time_data['total_time']) if 'total_time' in time_data else 0
            core_time = baseline_time + injection_time
            detection_overhead = total_time - core_time
            detection_overhead_pct = (detection_overhead / core_time * 100) if core_time > 0 else 0
            
            categories = ['Injection\nTime', 'Injection\nMemory', 'Detection\nTime']
            values = [time_overhead_pct, mem_overhead_pct, detection_overhead_pct]
            colors_bar = ['steelblue', 'coral', 'orange']
            
            bars = ax4.bar(categories, values, color=colors_bar, alpha=0.8)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:+.1f}%',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')
            
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax4.set_ylabel('Overhead (%)')
            ax4.set_title('Performance Overhead Summary')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            save_path = exp['exp_dir'] / "performance_impact_visualization.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def visualize_performance_sweep(self, sweep_dir: str, save_path: Optional[str] = None):
        """可视化参数扫描中的性能影响趋势"""
        sweep_path = self.results_dir / sweep_dir
        
        if not sweep_path.exists():
            print(f"Sweep directory not found: {sweep_path}")
            return
        
        # 加载所有实验
        sweep_experiments = []
        for exp_dir in sweep_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'violations':
                exp_id = exp_dir.name
                if exp_id not in self.experiments:
                    self.load_experiment(exp_id, parent_dir=sweep_path)
                if exp_id in self.experiments:
                    sweep_experiments.append(exp_id)
        
        print(f"Loaded {len(sweep_experiments)} sweep configurations")
        
        # 收集性能影响数据
        impact_by_bit = defaultdict(lambda: defaultdict(list))
        impact_by_location = defaultdict(lambda: defaultdict(list))
        
        for exp_id in sweep_experiments:
            exp = self.experiments[exp_id]
            config = exp.get('config')
            if not config:
                config = self.parse_config_from_exp_id(exp_id)
            
            if not config:
                continue
            
            results = exp.get('results', [])
            
            # 收集性能数据
            time_data = defaultdict(list)
            memory_data = defaultdict(list)
            
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                perf = result.get('extra_data', {}).get('performance', {})
                
                if 'time' in perf:
                    for key, value in perf['time'].items():
                        if value is not None and not np.isnan(value):
                            time_data[key].append(value)
                
                if 'memory' in perf:
                    for key, value in perf['memory'].items():
                        if value is not None and not np.isnan(value):
                            memory_data[key].append(value)
            
            # 计算影响指标
            bit = config.get('injection_bit') or config.get('bit')
            location = config.get('injection_location') or config.get('location')
            
            # 注入时间开销
            if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
                baseline_t = np.mean(time_data['baseline_forward_time'])
                injection_t = np.mean(time_data['injection_forward_time'])
                time_oh_pct = ((injection_t - baseline_t) / baseline_t * 100) if baseline_t > 0 else 0
                
                if bit is not None:
                    impact_by_bit[bit]['injection_time_oh'].append(time_oh_pct)
                if location:
                    impact_by_location[location]['injection_time_oh'].append(time_oh_pct)
            
            # 注入内存开销
            if 'baseline_memory_mb' in memory_data and 'injection_memory_mb' in memory_data:
                baseline_m = np.mean(memory_data['baseline_memory_mb'])
                injection_m = np.mean(memory_data['injection_memory_mb'])
                mem_oh_pct = ((injection_m - baseline_m) / baseline_m * 100) if baseline_m > 0 else 0
                
                if bit is not None:
                    impact_by_bit[bit]['injection_mem_oh'].append(mem_oh_pct)
                if location:
                    impact_by_location[location]['injection_mem_oh'].append(mem_oh_pct)
            
            # 检测时间占比
            if 'total_time' in time_data:
                total_t = np.mean(time_data['total_time'])
                bounds_t = np.mean(time_data['bounds_computation_time']) if 'bounds_computation_time' in time_data else 0
                violation_t = np.mean(time_data['violation_detection_time']) if 'violation_detection_time' in time_data else 0
                detection_t = bounds_t + violation_t
                detection_pct = (detection_t / total_t * 100) if total_t > 0 else 0
                
                if bit is not None:
                    impact_by_bit[bit]['detection_pct'].append(detection_pct)
                if location:
                    impact_by_location[location]['detection_pct'].append(detection_pct)
        
        if not impact_by_bit and not impact_by_location:
            print("No performance data available for visualization")
            return
        
        # 创建图表 (2x2 布局)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Performance Impact Sweep: {sweep_dir}', fontsize=14, fontweight='bold')
        
        # 1. 注入时间开销 vs 比特位
        ax1 = fig.add_subplot(gs[0, 0])
        
        if impact_by_bit:
            bits = sorted(impact_by_bit.keys())
            avg_time_oh = np.array([np.mean(impact_by_bit[b]['injection_time_oh']) if impact_by_bit[b]['injection_time_oh'] else 0 
                          for b in bits])
            
            ax1.plot(bits, avg_time_oh, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Bit Position')
            ax1.set_ylabel('Time Overhead (%)')
            ax1.set_title('Fault Injection Time Overhead by Bit')
            ax1.grid(True, alpha=0.3)
            
            # 标注符号位
            # if 31 in bits:
            #     ax1.axvline(x=31, color='r', linestyle='--', alpha=0.5)
            #     max_y = max(avg_time_oh) if avg_time_oh else 1
            #     ax1.text(31, max_y * 0.9, 'Sign bit', ha='center', fontsize=9, color='r')
        
        # 2. 注入内存开销 vs 比特位
        ax2 = fig.add_subplot(gs[0, 1])
        
        if impact_by_bit:
            bits = sorted(impact_by_bit.keys())
            avg_mem_oh = np.array([np.mean(impact_by_bit[b]['injection_mem_oh']) if impact_by_bit[b]['injection_mem_oh'] else 0 
                         for b in bits])
            
            ax2.plot(bits, avg_mem_oh, 's-', linewidth=2, markersize=8, color='coral')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Bit Position')
            ax2.set_ylabel('Memory Overhead (%)')
            ax2.set_title('Fault Injection Memory Overhead by Bit')
            ax2.grid(True, alpha=0.3)
        
        # 3. 检测开销占比 vs 位置（箱线图）
        ax3 = fig.add_subplot(gs[1, 0])
        
        if impact_by_location:
            locations = sorted(impact_by_location.keys())
            detection_data = [impact_by_location[loc]['detection_pct'] 
                            for loc in locations if impact_by_location[loc]['detection_pct']]
            
            if detection_data:
                bp = ax3.boxplot(detection_data, patch_artist=True)
                ax3.set_xticks(range(1, len(locations) + 1))
                ax3.set_xticklabels(locations)
            
                # 美化箱线图
                colors = colormaps['Set3'](np.linspace(0, 1, len(detection_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax3.set_ylabel('Detection Overhead (%)')
                ax3.set_title('Bounds Detection Overhead by Location')
                ax3.grid(True, alpha=0.3, axis='y')
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. 综合开销对比（分组柱状图）
        ax4 = fig.add_subplot(gs[1, 1])
        
        if impact_by_location:
            locations = sorted(impact_by_location.keys())
            
            avg_injection_time = np.array([np.mean(impact_by_location[loc]['injection_time_oh']) 
                                 if impact_by_location[loc]['injection_time_oh'] else 0
                                 for loc in locations])
            avg_injection_mem = np.array([np.mean(impact_by_location[loc]['injection_mem_oh'])
                                if impact_by_location[loc]['injection_mem_oh'] else 0
                                for loc in locations])
            avg_detection = np.array([np.mean(impact_by_location[loc]['detection_pct'])
                            if impact_by_location[loc]['detection_pct'] else 0
                            for loc in locations])
            
            x = np.arange(len(locations))
            width = 0.25
            
            bars1 = ax4.bar(x - width, avg_injection_time, width, label='Injection Time OH', 
                          color='steelblue', alpha=0.8)
            bars2 = ax4.bar(x, avg_injection_mem, width, label='Injection Mem OH',
                          color='coral', alpha=0.8)
            bars3 = ax4.bar(x + width, avg_detection, width, label='Detection OH',
                          color='orange', alpha=0.8)
            
            ax4.set_ylabel('Overhead (%)')
            ax4.set_title('Performance Overhead Comparison by Location')
            ax4.set_xticks(x)
            ax4.set_xticklabels(locations, rotation=45, ha='right')
            ax4.legend()
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            save_path = str(sweep_path / "performance_impact_sweep_visualization.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def compare_multiple_sweeps(self, sweep_configs: dict, save_dir: Optional[str] = None):
        """
        比较多个sweep实验的准确度和性能
        
        参数:
        sweep_configs: dict，格式为 {'k!v_s@w': 'sweep_dir1', 'k=v_s@w': 'sweep_dir2', ...}
        save_dir: 保存图表的目录，如果为None则保存到第一个sweep目录
        
        生成:
        1. 4个准确度热力图（每个参数vs bit position，包含4种方法）
        2. 1个性能对比图（4种方法的3条折线：injection time/mem/detection）
        """
        
        if not sweep_configs:
            print("No sweep configurations provided")
            return
        
        print(f"Comparing {len(sweep_configs)} sweep experiments...")
        
        # 为每个sweep加载数据
        all_sweep_data = {}
        
        for method_name, sweep_dir in sweep_configs.items():
            sweep_path = self.results_dir / sweep_dir
            
            if not sweep_path.exists():
                print(f"Sweep directory not found: {sweep_path}")
                continue
            
            # 为该方法创建独立的实验存储
            method_experiments = {}
            
            # 加载实验
            for exp_dir in sweep_path.iterdir():
                if exp_dir.is_dir() and exp_dir.name != 'violations':
                    exp_id = exp_dir.name
                    
                    # 强制重新加载（即使已存在也要重新加载，因为不同sweep的同名实验数据不同）
                    self.load_experiment(exp_id, parent_dir=sweep_path)
                    
                    # 深拷贝到该方法的独立存储（确保完全独立）
                    if exp_id in self.experiments:
                        method_experiments[exp_id] = copy.deepcopy(self.experiments[exp_id])
            
            print(f"  {method_name}: {len(method_experiments)} experiments")
            
            all_sweep_data[method_name] = {
                'sweep_path': sweep_path,
                'experiments': method_experiments  # 存储实验数据而非ID列表
            }
        
        if not all_sweep_data:
            print("No valid sweep data found")
            return
        
        # ===== 1. 准确度分析 - 生成4个热力图 =====
        parameters = ['injection_layers', 'injection_location', 'seed', 'injection_idx']
        param_names = ['Injection Layer', 'Injection Location', 'Random Seed', 'Injection Index']
        
        for param, param_name in zip(parameters, param_names):
            self._generate_accuracy_heatmap_comparison(all_sweep_data, param, param_name, save_dir)
        
        # ===== 2. 性能分析 - 生成对比折线图 =====
        self._generate_performance_comparison(all_sweep_data, save_dir)
        
        print("\nComparison complete!")
    
    def _generate_accuracy_heatmap_comparison(self, all_sweep_data: dict, param: str, param_name: str, save_dir: Optional[str]):
        """生成单个参数的准确度热力图对比（2x2布局，4种方法）"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
        
        fig.suptitle(f'Detection Rate Heatmap: {param_name} × Bit Position', fontsize=16, fontweight='bold')
        
        method_names = list(all_sweep_data.keys())
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for idx, (method_name, pos) in enumerate(zip(method_names, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            
            sweep_data = all_sweep_data[method_name]
            
            # 收集该方法的数据
            heatmap_data = defaultdict(lambda: defaultdict(list))
            
            # 直接遍历该方法的实验数据（已独立存储）
            for exp_id, exp in sweep_data['experiments'].items():
                config = exp.get('config')
                if not config:
                    config = self.parse_config_from_exp_id(exp_id)
                
                if not config:
                    continue
                
                # 获取参数值
                if param == 'injection_layers':
                    param_val = str(config.get('injection_layers') or config.get('layers', ''))
                elif param == 'injection_location':
                    param_val = config.get('injection_location') or config.get('location', 'unknown')
                elif param == 'seed':
                    param_val = config.get('seed', 0)
                elif param == 'injection_idx':
                    param_val = str(config.get('injection_idx') or config.get('idx', ''))
                else:
                    continue
                
                bit = config.get('injection_bit') or config.get('bit')
                if bit is None:
                    continue
                
                # 获取检测率
                summary = exp.get('summary')
                detection_rate = 0
                
                if summary and 'violation_detection' in summary:
                    detection_rate = summary['violation_detection'].get('detection_rate', 0)
                elif exp.get('results'):
                    results = exp.get('results', [])
                    violations = [r.get('violation_detected', False) for r in results if isinstance(r, dict)]
                    if violations:
                        detection_rate = sum(violations) / len(violations)
                
                # 跳过NaN
                if np.isnan(detection_rate):
                    continue
                
                heatmap_data[param_val][bit].append(detection_rate)
            
            # 构建热力图矩阵
            if not heatmap_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name}')
                continue
            
            param_values = sorted(heatmap_data.keys())
            bits = sorted(set(bit for param_dict in heatmap_data.values() for bit in param_dict.keys()))
            
            # 限制参数值数量（如果太多）
            if len(param_values) > 20:
                param_values = param_values[:20]
            
            matrix = np.zeros((len(param_values), len(bits)))
            
            for i, pval in enumerate(param_values):
                for j, bit in enumerate(bits):
                    if bit in heatmap_data[pval]:
                        matrix[i, j] = np.mean(heatmap_data[pval][bit]) * 100  # 转为百分比
                    else:
                        matrix[i, j] = np.nan
            
            # 绘制热力图
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            ax.set_xticks(np.arange(len(bits)))
            ax.set_yticks(np.arange(len(param_values)))
            ax.set_xticklabels(bits, fontsize=8)
            
            # 参数值标签处理
            if param in ['injection_layers', 'injection_idx']:
                labels = [str(pv)[:15] for pv in param_values]  # 截断长字符串
            else:
                labels = param_values
            ax.set_yticklabels(labels, fontsize=9)
            
            ax.set_xlabel('Bit Position', fontsize=10)
            ax.set_ylabel(param_name, fontsize=10)
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            
            # 添加colorbar
            plt.colorbar(im, ax=ax, label='Detection Rate (%)')
        
        plt.tight_layout()
        
        # 保存
        if save_dir:
            save_path = Path(save_dir) / f'comparison_heatmap_{param}.png'
        else:
            first_sweep = list(all_sweep_data.values())[0]['sweep_path']
            save_path = first_sweep.parent / f'comparison_heatmap_{param}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
        plt.close()
    
    def _generate_performance_comparison(self, all_sweep_data: dict, save_dir: Optional[str]):
        """生成性能对比折线图（2x2布局，每个方法3条折线）"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
        
        fig.suptitle('Performance Overhead Comparison by Bit Position', fontsize=16, fontweight='bold')
        
        method_names = list(all_sweep_data.keys())
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for idx, (method_name, pos) in enumerate(zip(method_names, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            
            sweep_data = all_sweep_data[method_name]
            
            # 收集性能数据
            perf_by_bit = defaultdict(lambda: defaultdict(list))
            
            # 直接遍历该方法的实验数据（已独立存储）
            for exp_id, exp in sweep_data['experiments'].items():
                config = exp.get('config')
                if not config:
                    config = self.parse_config_from_exp_id(exp_id)
                
                if not config:
                    continue
                
                bit = config.get('injection_bit') or config.get('bit')
                if bit is None:
                    continue
                
                results = exp.get('results', [])
                
                # 收集性能数据
                time_data = defaultdict(list)
                memory_data = defaultdict(list)
                
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    
                    perf = result.get('extra_data', {}).get('performance', {})
                    
                    if 'time' in perf:
                        for key, value in perf['time'].items():
                            if value is not None and not np.isnan(value):
                                time_data[key].append(value)
                    
                    if 'memory' in perf:
                        for key, value in perf['memory'].items():
                            if value is not None and not np.isnan(value):
                                memory_data[key].append(value)
                
                # 计算开销
                if 'baseline_forward_time' in time_data and 'injection_forward_time' in time_data:
                    baseline_t = np.mean(time_data['baseline_forward_time'])
                    injection_t = np.mean(time_data['injection_forward_time'])
                    time_oh = ((injection_t - baseline_t) / baseline_t * 100) if baseline_t > 0 else 0
                    perf_by_bit[bit]['injection_time_oh'].append(time_oh)
                
                if 'baseline_memory_mb' in memory_data and 'injection_memory_mb' in memory_data:
                    baseline_m = np.mean(memory_data['baseline_memory_mb'])
                    injection_m = np.mean(memory_data['injection_memory_mb'])
                    mem_oh = ((injection_m - baseline_m) / baseline_m * 100) if baseline_m > 0 else 0
                    perf_by_bit[bit]['injection_mem_oh'].append(mem_oh)
                
                if 'total_time' in time_data:
                    total_t = np.mean(time_data['total_time'])
                    bounds_t = np.mean(time_data['bounds_computation_time']) if 'bounds_computation_time' in time_data else 0
                    violation_t = np.mean(time_data['violation_detection_time']) if 'violation_detection_time' in time_data else 0
                    detection_t = bounds_t + violation_t
                    detection_pct = (detection_t / total_t * 100) if total_t > 0 else 0
                    perf_by_bit[bit]['detection_pct'].append(detection_pct)
            
            # 绘制折线图
            if not perf_by_bit:
                ax.text(0.5, 0.5, 'No performance data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name}')
                continue
            
            bits = sorted(perf_by_bit.keys())
            
            # 计算平均值
            injection_time_oh = np.array([np.mean(perf_by_bit[b]['injection_time_oh']) if perf_by_bit[b]['injection_time_oh'] else 0 
                                for b in bits])
            injection_mem_oh = np.array([np.mean(perf_by_bit[b]['injection_mem_oh']) if perf_by_bit[b]['injection_mem_oh'] else 0 
                               for b in bits])
            detection_pct = np.array([np.mean(perf_by_bit[b]['detection_pct']) if perf_by_bit[b]['detection_pct'] else 0 
                            for b in bits])
            
            # 绘制3条折线
            ax.plot(bits, injection_time_oh, 'o-', linewidth=2, markersize=6, 
                   label='Injection Time OH', color='steelblue', alpha=0.8)
            ax.plot(bits, injection_mem_oh, 's-', linewidth=2, markersize=6, 
                   label='Injection Memory OH', color='coral', alpha=0.8)
            ax.plot(bits, detection_pct, '^-', linewidth=2, markersize=6, 
                   label='Detection OH', color='orange', alpha=0.8)
            
            # 添加0线
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # 标注符号位
            if 31 in bits:
                ax.axvline(x=31, color='red', linestyle='--', alpha=0.3, linewidth=1)
                y_max = max(max(injection_time_oh), max(injection_mem_oh), max(detection_pct))
                ax.text(31, y_max * 0.95, 'Sign bit', ha='center', fontsize=8, color='red')
            
            ax.set_xlabel('Bit Position', fontsize=10)
            ax.set_ylabel('Overhead (%)', fontsize=10)
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_dir:
            save_path = Path(save_dir) / 'comparison_performance_overhead.png'
        else:
            first_sweep = list(all_sweep_data.values())[0]['sweep_path']
            save_path = first_sweep.parent / 'comparison_performance_overhead.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
        plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('results_dir', type=str, help='Results directory path')
    parser.add_argument('--exp-id', type=str, help='Experiment ID to visualize')
    parser.add_argument('--sweep', type=str, help='Parameter sweep directory to visualize')
    parser.add_argument('--epsilon-detail', nargs=2, metavar=('EXP_ID', 'RUN_ID'),
                       help='Visualize epsilon details for specific run')
    parser.add_argument('--perf', type=str, metavar='EXP_ID',
                       help='Visualize performance for a single experiment')
    parser.add_argument('--perf-sweep', type=str, metavar='SWEEP_DIR',
                       help='Visualize performance for a parameter sweep')
    parser.add_argument('--compare-sweeps', nargs='+', metavar='METHOD:SWEEP_DIR',
                       help='Compare multiple sweeps (format: method1:dir1 method2:dir2 ...)')
    parser.add_argument('--save', type=str, help='Save path for the figure')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results_dir)
    
    if args.exp_id:
        # 可视化单个实验
        visualizer.visualize_single_experiment(args.exp_id, args.save)
    
    elif args.sweep:
        # 可视化参数扫描
        visualizer.visualize_parameter_sweep(args.sweep, args.save)
    
    elif args.epsilon_detail:
        # 可视化epsilon详情
        exp_id, run_id = args.epsilon_detail
        visualizer.visualize_epsilon_details(exp_id, int(run_id), args.save)
    
    elif args.perf:
        # 可视化单个实验的性能
        visualizer.visualize_performance_single(args.perf, args.save)
    
    elif args.perf_sweep:
        # 可视化参数扫描的性能
        visualizer.visualize_performance_sweep(args.perf_sweep, args.save)
    
    elif args.compare_sweeps:
        # 比较多个sweep实验
        sweep_configs = {}
        for item in args.compare_sweeps:
            if ':' in item:
                method, sweep_dir = item.split(':', 1)
                sweep_configs[method] = sweep_dir
            else:
                print(f"Invalid format: {item}. Expected METHOD:SWEEP_DIR")
                return
        
        if sweep_configs:
            visualizer.compare_multiple_sweeps(sweep_configs, args.save)
        else:
            print("No valid sweep configurations provided")
    
    else:
        print("Please specify --exp-id, --sweep, --epsilon-detail, --perf, --perf-sweep, or --compare-sweeps")
        print("\nExamples:")
        print("  python visualize_results.py ./results --exp-id gpt2_simple_test_20241212_123456")
        print("  python visualize_results.py ./results --sweep gpt2_bit_sweep")
        print("  python visualize_results.py ./results --epsilon-detail gpt2_test 0")
        print("  python visualize_results.py ./results --perf gpt2_simple_test_20241212_123456")
        print("  python visualize_results.py ./results --perf-sweep gpt2_bit_sweep")
        print("  python visualize_results.py ./results --compare-sweeps k!v_s@w:sweep1 k=v_s@w:sweep2 k=v_q@o:sweep3 k=v_comb:sweep4")


if __name__ == "__main__":
    main()