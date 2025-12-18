"""
完整的实验运行器 - 集成所有组件的完整实现
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from pathlib import Path

from .fault_injection import InjectionConfig, InjectionLocation
from .bounds_computation import compute_attention_bounds, detect_violation, compute_injected_epsilon, compute_injected_epsilon_from_p
from .performance_monitor import PerformanceMonitor
from .experiment_runner import ExperimentResult, ResultsLogger


class IntermediateTensorCollector:
    """
    中间张量收集器
    
    职责：
    1. 从adapter返回的intermediates中收集张量
    2. 存储所有层的张量副本（clone）
    3. 提供统一的访问接口
    
    工作方式：
    - AttentionHook在forward中提供张量引用
    - IntermediateTensorCollector负责clone和存储
    """
    
    def __init__(self):
        self.tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self.enabled = False
    
    def enable(self):
        """启用收集"""
        self.enabled = True
        
    def disable(self):
        """禁用收集"""
        self.enabled = False
    
    def reset(self):
        """重置收集器"""
        self.tensors = {}
    
    def collect_from_intermediates(self, intermediates: Dict[str, Any]):
        """
        从adapter返回的intermediates中收集张量
        
        Args:
            intermediates: adapter返回的字典，包含:
                - 'layer_idx': 层索引
                - 'q', 'k', 'v', 'scores', 'weights', 'out': 张量引用
                - 'injection_applied': 是否应用了注入
        """
        if not self.enabled:
            return
        
        layer_idx = intermediates.get('layer_idx')
        if layer_idx is None:
            return
        
        if layer_idx not in self.tensors:
            self.tensors[layer_idx] = {}
        
        # 收集并clone所有张量
        for name, value in intermediates.items():
            if isinstance(value, torch.Tensor):
                # Clone以保存副本
                self.tensors[layer_idx][name] = value.detach().clone()
            elif name in ['layer_idx', 'injection_applied']:
                # 保存元信息
                self.tensors[layer_idx][name] = value
    
    def get_layer(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """获取某一层的所有张量"""
        return self.tensors.get(layer_idx)
    
    def get_all(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """获取所有张量"""
        return self.tensors


def setup_injection_for_model(model, injection_config: Optional[InjectionConfig]):
    """
    为模型设置注入配置
    
    Args:
        model: 已经过monkey patch的模型
        injection_config: 注入配置（None表示baseline）
    """
    if hasattr(model, 'transformer'):  # GPT-2
        for layer in model.transformer.h:
            if hasattr(layer.attn, 'adapter'):
                # 设置注入配置
                layer.attn._injection_config = injection_config


def run_model_with_collection(model, input_ids, attention_mask, 
                               collector: IntermediateTensorCollector,
                               injection_config: Optional[InjectionConfig] = None):
    """
    运行模型并收集中间张量
    
    这个函数临时替换forward方法来：
    1. 使用adapter进行前向传播
    2. 收集中间张量
    3. 应用错误注入（如果配置了）
    
    完成后恢复原始的forward方法
    
    Args:
        model: 模型（需要已经通过monkey_patch_model添加adapter）
        input_ids: 输入token ids
        attention_mask: 注意力掩码
        collector: 张量收集器
        injection_config: 注入配置（如果为None则是baseline）
        
    Returns:
        outputs: 模型输出
        collected: 收集到的张量字典
    """
    collector.reset()
    collector.enable()
    
    # 设置注入配置
    setup_injection_for_model(model, injection_config)
    
    # 保存原始的forward方法，以便恢复
    original_forwards = {}
    
    try:
        # 临时修改每层的forward以收集intermediates
        if hasattr(model, 'transformer'):  # GPT-2
            for layer in model.transformer.h:
                if hasattr(layer.attn, 'adapter') and hasattr(layer.attn, 'layer_idx'):
                    layer_idx = layer.attn.layer_idx
                    
                    # 保存真正的原始forward（应该在monkey_patch时保存了）
                    if hasattr(layer.attn, '_original_forward'):
                        original_forwards[layer_idx] = layer.attn._original_forward
                    else:
                        # Fallback: 保存当前的forward
                        original_forwards[layer_idx] = layer.attn.forward
                    
                    # 创建新的wrapper（使用默认参数捕获变量）
                    def make_forward_wrapper(attn_obj, adapter_obj, l_idx, col):
                        """创建forward wrapper，正确捕获所有变量"""
                        def wrapper(hidden_states, *args, **kwargs):
                            # 获取注入配置
                            inj_cfg = getattr(attn_obj, '_injection_config', None)
                            
                            # 调用adapter的forward_with_injection
                            output, intermediates = adapter_obj.forward_with_injection(
                                hidden_states,
                                layer_idx=l_idx,
                                injection_config=inj_cfg,
                                return_intermediates=True,
                                *args,
                                **kwargs
                            )
                            
                            # 收集中间张量
                            if col.enabled:
                                col.collect_from_intermediates(intermediates)
                            
                            # 返回output和attn_weights以兼容GPT2 API
                            # GPT2期望 (output, attn_weights) 或只返回output
                            # 我们返回(output, weights)如果有，否则只返回output
                            if 'weights' in intermediates:
                                return output, intermediates['weights']
                            else:
                                return output, None
                            
                            ###### 或者修改adapter让其返回attn_weights
                                                    
                        return wrapper
                    
                    # 设置新的forward（正确传递变量）
                    layer.attn.forward = make_forward_wrapper(
                        layer.attn,  # 传递attn对象
                        layer.attn.adapter,  # 传递adapter对象
                        layer_idx,  # 传递layer_idx
                        collector  # 传递collector
                    )
        
        # Forward模型
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    
    finally:
        # 恢复原始forward方法
        if hasattr(model, 'transformer'):
            for layer in model.transformer.h:
                if hasattr(layer.attn, 'layer_idx'):
                    layer_idx = layer.attn.layer_idx
                    if layer_idx in original_forwards:
                        layer.attn.forward = original_forwards[layer_idx]
        
        collector.disable()
    
    return outputs, collector.get_all()


class CompleteExperimentRunner:
    """完整的实验运行器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = ResultsLogger(config.save_dir, config.exp_id)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        self.logger.log(f"Experiment initialized: {config.exp_id}")
        self.logger.log(f"Device: {self.device}, Dtype: {self.dtype}")
        
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def run_single(self, run_id: int, model, dataloader) -> ExperimentResult:
        """
        运行单次完整实验
        
        Args:
            run_id: 运行ID
            model: 已经过monkey patch的模型
            dataloader: 数据加载器
            
        Returns:
            ExperimentResult
        """
        self.logger.log(f"Starting run {run_id}")
        
        # 初始化
        monitor = PerformanceMonitor(self.device)
        collector = IntermediateTensorCollector()
        
        result = ExperimentResult(
            exp_id=self.config.exp_id,
            run_id=run_id
        )
        
        # 获取数据
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            self.logger.log("No data in dataloader!", level="ERROR")
            return result
        
        # 准备输入
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(self.device)
            # attention_mask = batch.get('attention_mask', None)
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(self.device)
            attention_mask = None  # 目前不使用attention_mask
            labels = batch.get('labels', None)
            if labels is not None:
                labels = labels.to(self.device)
        else:
            input_ids = batch[0].to(self.device)
            attention_mask = None
            labels = batch[1].to(self.device) if len(batch) > 1 else None
        
        batch_size, seq_length = input_ids.shape
        monitor.record_model_info(model, batch_size, seq_length)
        
        ##################################################
        ##################################################
        # from utils.debug import debug
        # debug(attention_mask)
        ##################################################
        ##################################################
        
        with monitor.timer('total'):
            # ==================== Baseline ====================
            self.logger.log("Running baseline...")
            monitor.record_memory('before_baseline')
            
            model.eval()
            
            with monitor.timer('baseline_forward'):
                baseline_outputs, baseline_tensors = run_model_with_collection(
                    model, input_ids, attention_mask, collector, injection_config=None
                )
            
            monitor.record_memory('baseline')
            
            ##################################################
            ##################################################
            # from utils.debug import debug
            # if len(baseline_tensors) > 0:
            #     first_layer = list(baseline_tensors.keys())[0]
            #     debug(f"  Layer {first_layer} tensors: {list(baseline_tensors[first_layer].keys())}")
            ##################################################
            ##################################################
            
            # 计算loss
            if labels is not None:
                logits = baseline_outputs.logits
                baseline_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1),
                    ignore_index=-100  # 忽略padding
                ).item()
            elif hasattr(baseline_outputs, 'loss') and baseline_outputs.loss is not None:
                baseline_loss = baseline_outputs.loss.item()
            else:
                # 如果没有labels，尝试用input_ids作为targets
                try:
                    logits = baseline_outputs.logits
                    # 使用input_ids的shifted版本作为target
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    baseline_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    ).item()
                except:
                    baseline_loss = None
            
            result.baseline_loss = baseline_loss
            self.logger.log(f"Baseline loss: {baseline_loss}")
            
            # ==================== Injection ====================
            if self.config.injection_enabled:
                self.logger.log("Running with injection...")
                monitor.record_memory('before_injection')
                
                # 创建注入配置
                injection_cfg = InjectionConfig(
                    location=InjectionLocation(self.config.injection_location),
                    idx=self.config.injection_idx,
                    bit=self.config.injection_bit,
                    enabled=True
                )
                
                result.injection_info = injection_cfg.to_dict()
                
                with monitor.timer('injection_forward'):
                    injected_outputs, injected_tensors = run_model_with_collection(
                        model, input_ids, attention_mask, collector, injection_config=injection_cfg
                    )
                
                monitor.record_memory('injection')
                
                ##################################################
                ##################################################
                # from utils.debug import debug
                # if len(baseline_tensors) > 0:
                #     first_layer = list(injected_tensors.keys())[0]
                #     debug(f"  Layer {first_layer} tensors: {list(injected_tensors[first_layer].keys())}")
                ##################################################
                ##################################################
                
                # 计算loss
                if labels is not None:
                    logits = injected_outputs.logits
                    injected_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        labels.view(-1),
                        ignore_index=-100
                    ).item()
                elif hasattr(injected_outputs, 'loss') and injected_outputs.loss is not None:
                    injected_loss = injected_outputs.loss.item()
                else:
                    # Fallback: 使用shifted input_ids
                    try:
                        logits = injected_outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        injected_loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        ).item()
                    except:
                        injected_loss = None
                
                result.injected_loss = injected_loss
                
                if baseline_loss is not None and injected_loss is not None:
                    result.loss_diff = injected_loss - baseline_loss
                
                self.logger.log(f"Injected loss: {injected_loss}")
                self.logger.log(f"Loss diff: {result.loss_diff}")
                
                # ==================== Bounds Computation ====================
                self.logger.log("Computing bounds...")
                
                bounds_stats = {
                    'baseline': {},
                    'injected': {}
                }
                
                with monitor.timer('bounds_computation'):
                    # 对每个层计算边界
                    for layer_idx in baseline_tensors.keys():
                        baseline_layer = baseline_tensors[layer_idx]
                        
                        if 'scores' in baseline_layer and 'weights' in baseline_layer:
                            scores = baseline_layer['scores']
                            weights = baseline_layer['weights']
                            
                            # 获取head_dim
                            if 'q' in baseline_layer:
                                d = baseline_layer['q'].size(-1)
                            else:
                                d = 64  # 默认值
                            
                            # 计算边界
                            bounds = compute_attention_bounds(scores, weights, d)
                            
                            # 保存统计
                            bounds_stats['baseline'][layer_idx] = {
                                'epsilon_mean': bounds.epsilon.mean().item(),
                                'epsilon_std': bounds.epsilon.std().item(),
                                'lower_mean': bounds.lower1.mean().item(),
                                'upper_mean': bounds.upper.mean().item(),
                                'gamma_mean': bounds.gamma.mean().item(),
                            }
                
                # ==================== Violation Detection ====================
                self.logger.log("Detecting violations...")
                
                violation_detected = False
                total_violations = 0
                violation_positions = []
                epsilon_analysis_all = []
                
                with monitor.timer('violation_detection'):
                    for layer_idx in baseline_tensors.keys():
                        baseline_layer = baseline_tensors[layer_idx]
                        injected_layer = injected_tensors.get(layer_idx, {})
                        
                        if ('scores' in baseline_layer and 'weights' in baseline_layer and
                            'scores' in injected_layer and 'weights' in injected_layer and
                            'q' in baseline_layer and 'out' in injected_layer):
                            
                            # Baseline bounds
                            scores_b = baseline_layer['scores']
                            weights_b = baseline_layer['weights']
                            d = baseline_layer['q'].size(-1)
                            
                            bounds = compute_attention_bounds(scores_b, weights_b, d)
                            
                            # Injected epsilon
                            #############################################################
                            #############################################################
                            scores_i = injected_layer['scores']
                            injected_eps1, injected_eps2 = None, None
                            
                            if self.config.bound_type == "s@w" or self.config.bound_type == "comb":
                                weights_i = injected_layer['weights']
                            
                                injected_eps1 = compute_injected_epsilon_from_p(scores_i, weights_i, d)
                            #############################################################
                            #############################################################
                            if self.config.bound_type == "q@o" or self.config.bound_type == "comb":
                                attn_out_i = injected_layer['out']
                                q_b = baseline_layer['q']
                            
                                injected_eps2 = compute_injected_epsilon(scores_i, attn_out_i, q_b, d)
                            #############################################################
                            #############################################################
                            
                            # Detect violation
                            detection = detect_violation(bounds, injected_eps1, injected_eps2, self.config.tolerance)
                            
                            assert isinstance(detection['injection_violations'], Dict), "detection['injection_violations'] should be a dict"
                            
                            if detection['injection_violations']['any_violated']:
                                violation_detected = True
                                total_violations += detection['injection_violations']['num_upper_violations']
                                total_violations += detection['injection_violations']['num_lower_violations']
                                
                                if 'violation_positions' in detection:
                                    violation_positions.extend([
                                        (layer_idx, pos) for pos in detection['violation_positions']
                                    ])
                                    
                            if 'epsilon_analysis' in detection:
                                epsilon_analysis_all.append({
                                    'layer_idx': layer_idx,
                                    'analysis': detection['epsilon_analysis']
                                })
                
                result.violation_detected = violation_detected
                result.num_violations = total_violations
                result.violation_positions = violation_positions[:10]  # 只保存前10个
                
                result.baseline_epsilon_stats = bounds_stats.get('baseline')
                
                result.extra_data['epsilon_analysis'] = epsilon_analysis_all
                
                self.logger.log(f"Violations detected: {violation_detected}")
                self.logger.log(f"Total violations: {total_violations}")
        
        # 保存性能指标
        perf_metrics = monitor.get_metrics()
        result.extra_data['performance'] = perf_metrics.to_dict()
        
        monitor.print_summary()
        
        self.logger.log(f"Run {run_id} completed")
        
        return result
    
    def run_all(self, model, dataloader) -> List[ExperimentResult]:
        """运行所有重复实验"""
        self.logger.log(f"Starting experiment with {self.config.num_runs} runs")
        
        results = []
        for run_id in range(self.config.num_runs):
            result = self.run_single(run_id, model, dataloader)
            self.logger.add_result(result)
            results.append(result)
        
        # 保存汇总
        summary = self.logger.save_summary()
        self.logger.save_all_results()
        
        self.logger.log("Experiment completed")
        
        return results