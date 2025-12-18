from re import split
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, List, Any
from abc import ABC, abstractmethod


class AttentionHook:
    """
    注意力层Hook - 简化版本
    
    职责：
    1. 执行错误注入（如果配置了）
    2. 返回中间张量的引用（不存储副本）
    
    注意：张量的收集和存储由外部的IntermediateTensorCollector负责
    """
    
    def __init__(self, injection_config=None):
        self.injection_config = injection_config
        self.injection_applied = False
        # 不再存储captured_tensors，只是临时引用
        self._temp_tensors = {}
    
    def reset(self):
        """重置hook状态"""
        self._temp_tensors = {}
        self.injection_applied = False
    
    def register_tensor(self, name: str, tensor: torch.Tensor):
        """
        注册张量（不复制，只保存引用）
        
        这个方法只是在forward过程中临时保存引用，
        实际的收集由外部的IntermediateTensorCollector完成
        """
        self._temp_tensors[name] = tensor  # 只保存引用，不clone
    
    def get_tensors(self) -> Dict[str, Union[torch.Tensor, bool, int]]:
        """
        获取临时注册的张量引用
        
        注意：返回的是原始张量的引用，外部需要clone如果要保存
        """
        return self._temp_tensors
    
    def maybe_inject(self, name: str, tensor: torch.Tensor) -> bool:
        """
        可能执行注入
        
        Args:
            name: 张量名称 (q, k, v, scores, weights, out)
            tensor: 张量（会被原地修改）
            
        Returns:
            是否执行了注入
        """
        if self.injection_config is None:
            return False
        
        if not self.injection_config.enabled:
            return False
        
        if self.injection_config.location.value != name:
            return False
        
        # 执行注入
        from .fault_injection import bitflip_
        bitflip_(tensor, self.injection_config.idx, self.injection_config.bit)
        self.injection_applied = True
        
        return True


class BaseAttentionAdapter(ABC):
    """注意力层适配器基类"""
    
    @abstractmethod
    def forward_with_injection(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        injection_config = None,
        return_intermediates: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, bool, int]]]:
        """
        带注入的前向传播
        
        Args:
            hidden_states: 输入张量
            attention_mask: 注意力掩码
            injection_config: 注入配置
            return_intermediates: 是否返回中间张量
            
        Returns:
            output: 输出张量
            intermediates: 中间张量字典 {
                'q': query,
                'k': key,
                'v': value,
                'scores': attention_scores,
                'weights': attention_weights
            }
        """
        pass


class GPT2AttentionAdapter(BaseAttentionAdapter):
    """GPT-2注意力层适配器 - 兼容不同版本的transformers"""
    
    def __init__(self, original_attention):
        """
        Args:
            original_attention: transformers.models.gpt2.modeling_gpt2.GPT2Attention
        """
        self.attn = original_attention
        
        # 获取配置参数（兼容不同版本）
        self.num_heads = getattr(self.attn, 'num_heads', 12)
        self.head_dim = getattr(self.attn, 'head_dim', 64)
        self.split_size = getattr(self.attn, 'split_size', self.num_heads * self.head_dim)
        
        # 检查是否有scale属性
        self.scale_attn_weights = getattr(self.attn, 'scale_attn_weights', True)
        
        print(f"GPT2AttentionAdapter initialized: heads={self.num_heads}, head_dim={self.head_dim}")
    
    def _split_heads(self, tensor, num_heads, head_dim):
        """
        手动实现split_heads，不依赖原始模型的方法
        
        将 (batch, seq_len, hidden_size) -> (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_length = tensor.size()[:2]
        # Reshape: (batch, seq_len, hidden) -> (batch, seq_len, num_heads, head_dim)
        tensor = tensor.view(batch_size, seq_length, num_heads, head_dim)
        # Permute: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        return tensor.permute(0, 2, 1, 3)
    
    def _merge_heads(self, tensor, num_heads, head_dim):
        """
        手动实现merge_heads
        
        将 (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        """
        # Permute: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_length = tensor.size()[:2]
        # Reshape: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_size)
        return tensor.view(batch_size, seq_length, num_heads * head_dim)
    
    def forward_with_injection(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        injection_config = None,
        return_intermediates: bool = True,
        **kwargs
    ):
        """
        GPT-2注意力的修改版本
        
        完全手动实现，不依赖GPT2Attention的内部方法
        """
        
        hook = AttentionHook(injection_config)
        
        # 1. QKV投影
        # c_attn: (batch, seq_len, 3*hidden_size)
        qkv = self.attn.c_attn(hidden_states)
        
        # 分割成Q, K, V
        query, key, value = qkv.split(self.split_size, dim=2)
        
        # 2. Reshape for multi-head attention
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        # 注册张量（只保存引用，不复制）
        if return_intermediates:
            hook.register_tensor('q', query)
            hook.register_tensor('k', key)
            hook.register_tensor('v', value)
        
        # 可能注入到Q/K/V
        hook.maybe_inject('q', query)
        hook.maybe_inject('k', key)
        hook.maybe_inject('v', value)
        
        # 3. 计算attention scores
        # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
        # -> (batch, heads, seq_len, seq_len)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        # Scale
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, 
                dtype=attn_weights.dtype, 
                device=attn_weights.device
            )
        
        if return_intermediates:
            hook.register_tensor('scores', attn_weights)
        
        # 可能注入到scores
        hook.maybe_inject('scores', attn_weights)
        
        # 4. 应用attention mask（如果有）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 5. Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # 6. Dropout
        if hasattr(self.attn, 'attn_dropout'):
            attn_weights = self.attn.attn_dropout(attn_weights)
        
        if return_intermediates:
            hook.register_tensor('weights', attn_weights)
        
        # 可能注入到weights
        hook.maybe_inject('weights', attn_weights)
        
        # 7. Apply attention to values
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim)
        # -> (batch, heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value)
        
        # 可能注入到out
        hook.maybe_inject('out', attn_output)
        
        if return_intermediates:
            hook.register_tensor('out', attn_output)
        
        # 8. Merge heads
        # (batch, heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        # 9. Output projection
        attn_output = self.attn.c_proj(attn_output)
        
        # 10. Residual dropout
        if hasattr(self.attn, 'resid_dropout'):
            attn_output = self.attn.resid_dropout(attn_output)
        
        # 获取张量引用并添加元信息
        intermediates = hook.get_tensors()
        if layer_idx is not None:
            intermediates['layer_idx'] = layer_idx
        intermediates['injection_applied'] = hook.injection_applied
        
        return attn_output, intermediates


class DistilBertAttentionAdapter(BaseAttentionAdapter):
    """DistilBERT注意力层适配器"""
    
    def __init__(self, original_attention):
        """
        Args:
            original_attention: transformers.models.distilbert.modeling_distilbert.MultiHeadSelfAttention
        """
        self.attn = original_attention
    
    def forward_with_injection(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        injection_config = None,
        return_intermediates: bool = True,
        **kwargs
    ):
        """DistilBERT注意力的修改版本"""
        
        hook = AttentionHook(injection_config)
        
        bs, seq_len, dim = hidden_states.size()
        dim_per_head = self.attn.dim // self.attn.n_heads
        
        def shape(x):
            return x.view(bs, seq_len, self.attn.n_heads, dim_per_head).transpose(1, 2)
        
        def unshape(x):
            return x.transpose(1, 2).contiguous().view(bs, seq_len, self.attn.n_heads * dim_per_head)
        
        # QKV投影
        q = shape(self.attn.q_lin(hidden_states))
        k = shape(self.attn.k_lin(hidden_states))
        v = shape(self.attn.v_lin(hidden_states))
        
        if return_intermediates:
            hook.register_tensor('q', q)
            hook.register_tensor('k', k)
            hook.register_tensor('v', v)
        
        hook.maybe_inject('q', q)
        hook.maybe_inject('k', k)
        hook.maybe_inject('v', v)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / (dim_per_head ** 0.5)
        
        if return_intermediates:
            hook.register_tensor('scores', scores)
        
        hook.maybe_inject('scores', scores)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Attention weights
        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.attn.dropout(weights)
        
        if return_intermediates:
            hook.register_tensor('weights', weights)
        
        hook.maybe_inject('weights', weights)
        
        # Output
        context = torch.matmul(weights, v)
        
        hook.maybe_inject('out', context)
        
        context = unshape(context)
        context = self.attn.out_lin(context)
        
        return context, hook.get_tensors()


def create_adapter(model_type: str, attention_layer):
    """
    工厂函数:创建合适的适配器
    
    Args:
        model_type: 模型类型 ('gpt2', 'distilbert', etc.)
        attention_layer: 原始注意力层
        
    Returns:
        适配器对象
    """
    adapters = {
        'gpt2': GPT2AttentionAdapter,
        'distilbert': DistilBertAttentionAdapter,
        # 可以继续添加其他模型
    }
    
    adapter_class = adapters.get(model_type.lower())
    if adapter_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return adapter_class(attention_layer)


def monkey_patch_model(model, model_type: str, injection_layers: Optional[List[int]] = None,
                       force_kv_equal: bool = False):
    """
    对模型进行monkey patch - 只添加adapter，不替换forward
    
    这个函数只负责：
    1. 创建adapter对象
    2. 保存到layer.attn.adapter
    3. 记录layer_idx
    4. 初始化_injection_config
    
    实际的forward替换由run_model_with_collection负责（临时替换）
    
    Args:
        model: HuggingFace模型
        model_type: 模型类型
        injection_layers: 要注入的层索引列表，None表示所有层
        force_kv_equal: 是否强制K=V (通过共享权重)
        
    Returns:
        修改后的模型
    """
    
    if model_type.lower() == 'gpt2':
        # 检查和处理KV权重
        if force_kv_equal:
            print("Warning: Forcing K=V by weight sharing in GPT-2")
            print("This feature is experimental and may affect model performance!")
            # TODO: 实现权重共享逻辑
            force_kv_consistent(model, mode="k<-V")
        
        # 获取总层数
        total_layers = len(model.transformer.h)
        
        # 确定要patch的层
        if injection_layers is None:
            layers_to_patch = list(range(total_layers))
        else:
            layers_to_patch = [i for i in injection_layers if 0 <= i < total_layers]
        
        print(f"Patching {len(layers_to_patch)} out of {total_layers} attention layers")
        print(f"Layer indices: {layers_to_patch}")
        
        # 替换指定的GPT2Attention层
        for layer_idx in layers_to_patch:
            layer = model.transformer.h[layer_idx]
            original_attn = layer.attn
            
            # 创建adapter
            adapter = GPT2AttentionAdapter(original_attn)
            
            # 保存adapter和layer_idx到layer.attn
            layer.attn.adapter = adapter
            layer.attn.layer_idx = layer_idx
            layer.attn._injection_config = None  # 初始化注入配置
            layer.attn._original_forward = layer.attn.forward  # 保存原始forward
        
        print("✓ Model patching completed (adapters added, forward not replaced)")
    
    elif model_type.lower() == 'distilbert':
        total_layers = len(model.transformer.layer)
        
        if injection_layers is None:
            layers_to_patch = list(range(total_layers))
        else:
            layers_to_patch = [i for i in injection_layers if 0 <= i < total_layers]
        
        print(f"Patching {len(layers_to_patch)} out of {total_layers} attention layers")
        
        # 替换指定的DistilBERT attention层
        for layer_idx in layers_to_patch:
            layer = model.transformer.layer[layer_idx]
            original_attn = layer.attention
            adapter = DistilBertAttentionAdapter(original_attn)
            
            layer.attention.adapter = adapter
            layer.attention.layer_idx = layer_idx
            layer.attention._injection_config = None
            layer.attention._original_forward = layer.attention.forward
        
        print("✓ Model patching completed (adapters added, forward not replaced)")
    
    else:
        raise ValueError(f"Unsupported model type for patching: {model_type}")
    
    # 记录patch信息
    model._injection_config = {
        'model_type': model_type,
        'patched_layers': layers_to_patch,
        'force_kv_equal': force_kv_equal
    }
    
    return model


def check_kv_consistency(model, model_type: str) -> Dict[str, Any]:
    """
    检查模型的K和V权重是否一致
    
    Returns:
        检查结果字典
    """
    result = {
        'model_type': model_type,
        'kv_equal': False,
        'layer_info': []
    }
    
    if model_type.lower() == 'gpt2':
        for i, layer in enumerate(model.transformer.h):
            # GPT-2的c_attn包含QKV，需要检查K和V部分
            c_attn_weight = layer.attn.c_attn.weight
            split_size = layer.attn.split_size
            
            # 权重形状: [hidden, 3*hidden]
            # Q: [0:split], K: [split:2*split], V: [2*split:3*split]
            q_weight, k_weight, v_weight = c_attn_weight.split(split_size, dim=1)
            
            kv_equal = torch.allclose(k_weight, v_weight, rtol=1e-5)
            
            result['layer_info'].append({
                'layer_idx': i,
                'kv_equal': kv_equal,
                'k_shape': list(k_weight.shape),
                'v_shape': list(v_weight.shape)
            })
    
    # 检查是否所有层都相等
    result['kv_equal'] = all(info['kv_equal'] for info in result['layer_info'])
    
    return result

@torch.no_grad()
def force_kv_consistent(model, mode: str = "K<-V"):
    """
    强制所有层的 K、V 权重相等。
    mode: "K<-V" 表示令 K 等于 V；"V<-K" 表示令 V 等于 K
    """
    for layer in model.transformer.h:
        W = layer.attn.c_attn.weight
        b = layer.attn.c_attn.bias
        s = layer.attn.split_size

        # 找到 3*s 在哪一维：0 还是 1
        axis = 1 if W.shape[1] == 3 * s else 0

        def slc(t, i, j):
            # 返回 t 在 3*split 那一维上的 [i:j] 切片
            return t[:, i:j] if axis == 1 else t[i:j, :]

        if mode.upper() in ("K<-V", "K<=V"):
            # K 区间 [s:2s] 复制为 V 区间 [2s:3s]
            slc(W, s, 2*s).copy_(slc(W, 2*s, 3*s))
            if b is not None:
                b[s:2*s].copy_(b[2*s:3*s])
        elif mode.upper() in ("V<-K", "V<=K"):
            # V 区间 [2s:3s] 复制为 K 区间 [s:2s]
            slc(W, 2*s, 3*s).copy_(slc(W, s, 2*s))
            if b is not None:
                b[2*s:3*s].copy_(b[s:2*s])
        else:
            raise ValueError("mode must be 'K<-V' or 'V<-K'")