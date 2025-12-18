"""
边界计算模块 - 计算注意力机制的理论上下界
"""

import math
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class BoundsResult:
    """边界计算结果"""
    a_star: torch.Tensor  # 最大值
    w_star: torch.Tensor  # 最大权重
    gamma: torch.Tensor   # margin
    epsilon: torch.Tensor # 目标量
    lower1: torch.Tensor  # 下界1
    middle: torch.Tensor  # 中间值
    upper1: torch.Tensor  # 上界1
    upper2: torch.Tensor  # 上界2
    upper: torch.Tensor   # 最终上界
    
    # 额外信息
    valid_mask: Optional[torch.Tensor] = None  # 有效位置掩码(处理NaN)
    
    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        """转换为字典"""
        return {
            'a_star': self.a_star,
            'w_star': self.w_star,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'lower1': self.lower1,
            'middle': self.middle,
            'upper1': self.upper1,
            'upper2': self.upper2,
            'upper': self.upper,
            'valid_mask': self.valid_mask
        }
    
    def check_inequalities(self, eps: float = 1e-6) -> Dict[str, bool]:
        """
        检查不等式是否成立
        
        Returns:
            检查结果字典
        """
        if self.valid_mask is not None:
            # 只检查有效位置
            lower_ok = ((self.lower1 <= self.middle + eps) | ~self.valid_mask).all()
            mid_ok = ((self.middle <= self.epsilon + eps) | ~self.valid_mask).all()
            upper_ok = ((self.epsilon <= self.upper + eps) | ~self.valid_mask).all()
        else:
            lower_ok = (self.lower1 <= self.middle + eps).all()
            mid_ok = (self.middle <= self.epsilon + eps).all()
            upper_ok = (self.epsilon <= self.upper + eps).all()
        
        return {
            'lower1_le_middle': bool(lower_ok.item()),
            'middle_le_epsilon': bool(mid_ok.item()),
            'epsilon_le_upper': bool(upper_ok.item()),
            'all_valid': bool((lower_ok & mid_ok & upper_ok).item())
        }


@torch.no_grad()
def compute_attention_bounds(
    scores: torch.Tensor, 
    p: torch.Tensor, 
    d: int,
    handle_nan: bool = True,
    causal_mask: Optional[torch.Tensor] = None
) -> BoundsResult:
    """
    计算注意力机制的上下界
    
    Args:
        scores: 注意力分数张量 (B, H, T, T)
        p: 注意力权重张量 (B, H, T, T)
        d: 头维度
        handle_nan: 是否处理NaN值
        causal_mask: 因果掩码(可选)
        
    Returns:
        BoundsResult对象
    """
    B, H, T, _ = scores.shape
    device = scores.device
    dtype = scores.dtype
    sqrt_d = math.sqrt(d)
    n = T
    
    # 处理NaN和Inf
    if handle_nan:
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 创建有效位置掩码
        valid_mask = torch.isfinite(scores) & torch.isfinite(p)
        valid_mask = valid_mask.all(dim=-1)  # (B, H, T)
    else:
        valid_mask = None
    
    # 计算top-2值
    top2_vals, _ = torch.topk(scores, k=min(2, T), dim=-1)
    a_star = top2_vals[..., 0]  # (B, H, T)
    
    if T > 1:
        second = top2_vals[..., 1]
    else:
        second = a_star  # 只有一个值时,second等于a_star
    
    # 计算最大权重
    w_star = p.max(dim=-1).values  # (B, H, T)
    
    # 计算margin
    gamma = a_star - second
    
    # 计算期望和epsilon
    Ea = (p * scores).sum(dim=-1)  # (B, H, T)
    Ea = torch.nan_to_num(Ea, nan=0.0)
    epsilon = sqrt_d * (a_star - Ea)
    
    # 计算下界
    lower1 = sqrt_d * gamma / (1.0 + torch.exp(gamma))
    
    # 计算中间值
    middle = sqrt_d * gamma * (1.0 - w_star)
    
    # 计算上界1 (mean-based)
    upper1 = sqrt_d * (a_star - scores.mean(dim=-1))
    
    # 计算上界2 (Lambert W-based)
    lam_arg = torch.tensor((n - 1) / math.e, device=device, dtype=dtype)
    W_np = np.asarray(lambertw(lam_arg.detach().cpu().numpy(), 0).real)
    W = torch.as_tensor(W_np, device=device, dtype=dtype)
    
    cond = gamma >= (W + 1.0)
    term_case1 = sqrt_d * ((n - 1) * torch.exp(-gamma)) / \
                 (1.0 + (n - 1) * torch.exp(-gamma)) * gamma
    term_case2 = sqrt_d * W
    upper2 = torch.where(cond, term_case1, term_case2)
    
    # 最终上界
    upper = torch.minimum(upper1, upper2)
    
    return BoundsResult(
        a_star=a_star,
        w_star=w_star,
        gamma=gamma,
        epsilon=epsilon,
        lower1=lower1,
        middle=middle,
        upper1=upper1,
        upper2=upper2,
        upper=upper,
        valid_mask=valid_mask
    )


@torch.no_grad()
def compute_injected_epsilon(
    scores: torch.Tensor,
    attn_out: torch.Tensor,
    q: torch.Tensor,
    d: int
) -> torch.Tensor:
    """
    计算注入错误后的epsilon值
    
    Args:
        scores: 注意力分数 (B, H, T, T)
        attn_out: 注意力输出 (B, H, T, Dh)
        q: 查询张量 (B, H, T, Dh)
        d: 头维度
        
    Returns:
        epsilon值 (B, H, T)
    """
    sqrt_d = d ** 0.5
    a_star = scores.max(dim=-1).values  # (B, H, T)
    
    # 计算 <attn_out, q> 的内积
    Ea = (attn_out * q).sum(dim=-1)  # (B, H, T)
    
    return sqrt_d * a_star - Ea


@torch.no_grad()
def compute_injected_epsilon_from_p(
    scores: torch.Tensor,
    p: torch.Tensor,
    d: int
) -> torch.Tensor:
    """
    计算注入错误后的epsilon值（基于scores与p）

    Args:
        scores: 注意力分数 (B, H, T, T)
        p: 注意力概率 (B, H, T, T)
        d: 头维度

    Returns:
        epsilon值 (B, H, T)
    """
    sqrt_d = math.sqrt(d)
    a_star = scores.max(dim=-1).values       # (B, H, T)
    Ea = (p * scores).sum(dim=-1)            # (B, H, T)
    Ea = torch.nan_to_num(Ea, nan=0.0)
    return sqrt_d * (a_star - Ea)


def detect_violation(
    bounds: BoundsResult,
    injected_epsilon1: Optional[torch.Tensor] = None,
    injected_epsilon2: Optional[torch.Tensor] = None,
    tolerance: float = 0.0
) -> Dict[str, Union[Dict, List]]:
    """
    检测边界违反
    
    Args:
        bounds: 边界计算结果
        injected_epsilon: 注入错误后的epsilon(可选)
        tolerance: 容差
        
    Returns:
        检测结果字典
    """
    result: Dict[str, Union[Dict, List]] = {
        'baseline_violations': {},
        'injection_violations': {}
    }
    
    # 检查基线不等式
    baseline_check = bounds.check_inequalities(tolerance)
    result['baseline_violations'] = baseline_check
    
    base_false = torch.zeros_like(bounds.middle, dtype=torch.bool)
    lower1, lower2, upper1, upper2 = [base_false] * 4
    injected_epsilon = None
    
    if injected_epsilon1 is not None:
        injected_epsilon = injected_epsilon1
        lower1 = injected_epsilon1 < bounds.middle - tolerance
        upper1 = injected_epsilon1 > bounds.upper + tolerance
    
    if injected_epsilon2 is not None:
        injected_epsilon = injected_epsilon2
        lower2 = injected_epsilon2 < bounds.middle - tolerance
        upper2 = injected_epsilon2 > bounds.upper + tolerance
        
    # 如果提供了注入后的epsilon,检查是否违反边界
    if injected_epsilon is not None:
        lower_violation = lower1 | lower2
        upper_violation = upper1 | upper2
        epsilon_diff = injected_epsilon - bounds.epsilon
        
        if bounds.valid_mask is not None:
            lower_violation = lower_violation & bounds.valid_mask
            upper_violation = upper_violation & bounds.valid_mask
            epsilon_diff[~bounds.valid_mask] = 0.0
        
        top_k = min(20, epsilon_diff.numel())
        abs_flat_diff = torch.abs(epsilon_diff.flatten())
        top_values, top_indices = torch.topk(abs_flat_diff, k=top_k)
        
        shape = epsilon_diff.shape
        top_positions, epsilon_changes, boundary_info = [], [], []
        
        ##################################################
        ##################################################
        # from utils.debug import debug
        # debug(shape, bounds.epsilon.shape)
        ##################################################
        ##################################################
        
        for val, idx in zip(top_values, top_indices):
            if val.item() == 0.0:  # 跳过零值
                continue
                
            # 转换为多维索引
            multi_idx = []
            temp_idx = idx.item()
            for dim_size in reversed(shape):
                multi_idx.insert(0, temp_idx % dim_size)
                temp_idx = temp_idx // dim_size
            multi_idx = tuple(multi_idx)
            
            # 获取该位置的详细信息
            top_positions.append(multi_idx)
            epsilon_changes.append({
                'baseline_epsilon': bounds.epsilon[multi_idx].item(),
                'injected_epsilon': injected_epsilon[multi_idx].item(),
                'epsilon_diff': epsilon_diff[multi_idx].item(),
                'abs_diff': val.item()
            })
            boundary_info.append({
                'lower1': bounds.lower1[multi_idx].item(),
                'middle': bounds.middle[multi_idx].item(),
                'upper': bounds.upper[multi_idx].item(),
                'gamma': bounds.gamma[multi_idx].item(),
            })
            
            
        result['injection_violations'] = {
            'lower_violated': bool(lower_violation.any().item()),
            'upper_violated': bool(upper_violation.any().item()),
            'any_violated': bool((lower_violation | upper_violation).any().item()),
            'num_lower_violations': int(lower_violation.sum().item()),
            'num_upper_violations': int(upper_violation.sum().item())
        }
        
        # 记录违反位置
        if result['injection_violations']['any_violated']:
            violations = lower_violation | upper_violation
            positions = torch.nonzero(violations, as_tuple=False)
            result['violation_positions'] = positions.tolist()
            
        result['epsilon_analysis'] = {
            'mean_diff': epsilon_diff.mean().item(),
            'std_diff': epsilon_diff.std().item(),
            'max_abs_diff': abs_flat_diff.max().item(),
            'top_changes': list(zip(top_positions, epsilon_changes, boundary_info))
        }
    
    return result

# 导入scipy的Lambert W函数
try:
    from scipy.special import lambertw
except ImportError:
    print("Warning: scipy not available, Lambert W function will not work")
    def lambertw(z: Any, k: int = 0, tol: float = 1e-8) -> Any:
        raise NotImplementedError("scipy is required for Lambert W function")