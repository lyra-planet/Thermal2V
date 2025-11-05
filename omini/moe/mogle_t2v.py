import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.nn import functional as F


class Expert(nn.Module):
    """
    Expert - 用于热成像到可见光的跨模态特征转换
    
    处理Flux packed latent格式: [bs, seq_len, dim]
    其中 seq_len=256 (packed后), dim=64
    """
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 全连接层用于序列处理
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x):
        """
        x: (bs, seq_len, input_dim) - Flux packed latent format
        输出: (bs, seq_len, output_dim)
        """
        bs, seq_len, dim = x.shape
        # 展平为 (bs*seq_len, input_dim)
        x = x.view(-1, dim)
        # 通过网络
        x = self.net(x)
        # 恢复形状 (bs, seq_len, output_dim)
        x = x.view(bs, seq_len, -1)
        return x


class DynamicGatingNetwork(nn.Module):
    """
    动态门控网络 - 根据噪声和时间步动态调整权重
    """
    def __init__(self, hidden_dim=64, embed_dim=64, dtype=torch.bfloat16):
        super().__init__()
        
        self.time_proj = Timesteps(
            64, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedding = TimestepEmbedding(64, embed_dim)
        self.timestep_embedding = self.timestep_embedding.to(dtype=torch.bfloat16)
        
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dtype = dtype
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, thermal_features, noise_latent, timestep):
        """
        thermal_features: (bs, seq_len, hidden_dim) - 热成像特征
        noise_latent: (bs, seq_len, hidden_dim) - 噪声latent
        timestep: (bs,) - 时间步
        
        返回: (bs, seq_len, 1) - 动态权重
        """
        bs, seq_len, hidden_dim = thermal_features.shape
        
        # 时间步embedding
        time_emb = self.time_proj(timestep)  # (bs, 64)
        time_emb = time_emb.to(self.dtype)
        time_emb = self.timestep_embedding(time_emb)  # (bs, embed_dim)
        
        # 扩展到序列长度
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, embed_dim)
        
        # 噪声投影
        noise_emb = self.noise_proj(noise_latent)  # (bs, seq_len, hidden_dim)
        
        # 融合三个信息
        fused_input = thermal_features + noise_emb + time_emb  # (bs, seq_len, hidden_dim)
        
        # 生成权重
        weight = self.gate(fused_input)  # (bs, seq_len, 1)
        weight = torch.sigmoid(weight)
        
        return weight


class MoGLE(nn.Module):
    """
    MoGLE for Thermal-to-Visible Translation
    
    处理Flux pipeline的packed latent格式:
    - 输入: [bs, 256, 64] (packed latent)
    - 输出: [bs, 256, 64] (处理后的latent)
    
    工作流程:
    1. Expert: 热成像特征 → 跨模态转换特征
    2. DynamicGating: 根据噪声和时间步计算权重
    3. 加权融合: expert_output * weights
    """
    def __init__(
        self,
        input_dim=64,
        hidden_dim=256,
        output_dim=64,
        has_expert=True,
        has_gating=True,
        weight_is_scale=False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_expert = has_expert
        
        # Expert网络 - 学习热成像到可见光的转换
        if has_expert:
            self.expert = Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
        else:
            self.expert = nn.Identity()
        
        # 动态门控网络
        if has_gating:
            self.gating = DynamicGatingNetwork(hidden_dim=input_dim)
        else:
            self.gating = None
        
        self.weight_is_scale = weight_is_scale
    
    def forward(self, thermal_latent, noise_latent, timestep):
        """
        thermal_latent: (bs, seq_len, input_dim) - 热成像的packed latent
                        例如: (4, 256, 64)
        noise_latent: (bs, seq_len, input_dim) - 扩散过程中的噪声packed latent
                      例如: (4, 256, 64)
        timestep: (bs,) - 当前时间步
        
        返回: (bs, seq_len, output_dim) - 处理后的condition packed latent
        """
        # 通过Expert进行热成像→可见光的特征转换
        expert_output = self.expert(thermal_latent)  # (bs, seq_len, output_dim)
        
        # 如果没有gating，直接返回
        if self.gating is None:
            return expert_output
        
        # 计算动态权重
        weights = self.gating.forward(
            thermal_latent, 
            noise_latent=noise_latent, 
            timestep=timestep
        )  # (bs, seq_len, 1)
        
        # 加权融合
        output = expert_output * weights  # (bs, seq_len, output_dim)
        
        return output  # (bs, seq_len, output_dim)