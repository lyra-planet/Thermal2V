import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.nn import functional as F


class ResidualConvBlock(nn.Module):
    """残差卷积块 - 更深的网络中梯度流更好"""
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.act(out)
        return out


class ChannelAttention(nn.Module):
    """通道注意力 - 自适应学习重要特征"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 8), channels)
        )
    
    def forward(self, x):
        avg = self.avg_pool(x).squeeze(-1)
        max_val = self.max_pool(x).squeeze(-1)
        avg = self.fc(avg).unsqueeze(-1)
        max_val = self.fc(max_val).unsqueeze(-1)
        return x * torch.sigmoid(avg + max_val)


class SpatialAttention(nn.Module):
    """空间注意力 - 关注序列中的重要位置"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class CBAM(nn.Module):
    """通道和空间双注意力 - 结合两种注意力优势"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class UNetExpert(nn.Module):
    """
    改进的UNet Expert - 用于热成像到可见光的跨模态特征转换
    
    改进点:
    1. 残差连接 - 更好的梯度流
    2. CBAM注意力 - 自适应特征选择
    3. GELU激活 - 比ReLU更平滑
    4. 灵活的下采样策略 - 适应不同seq_len
    5. 特征融合加权 - 跳跃连接的权重融合
    """
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=64, num_downsample=2, use_attention=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_downsample = num_downsample
        self.use_attention = use_attention
        
        # 初始投影
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Encoder (下采样路径)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList() if use_attention else None
        self.downsample = nn.ModuleList()
        
        in_channels = hidden_dim
        for i in range(num_downsample):
            # 残差块
            self.encoder_blocks.append(ResidualConvBlock(in_channels, kernel_size=3))
            if use_attention:
                self.encoder_attentions.append(CBAM(in_channels))
            # 下采样
            self.downsample.append(nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1))
            in_channels *= 2
        
        # Bottleneck
        self.bottleneck = ResidualConvBlock(in_channels, kernel_size=3)
        if use_attention:
            self.bottleneck_attention = CBAM(in_channels)
        
        # Decoder (上采样路径)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList() if use_attention else None
        self.upsample = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i in range(num_downsample):
            out_channels = in_channels // 2
            # 上采样
            self.upsample.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            # 融合层 - 处理跳跃连接 (out_channels * 2因为会cat)
            self.fusion_layers.append(nn.Sequential(
                nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            ))
            # 残差块
            self.decoder_blocks.append(ResidualConvBlock(out_channels, kernel_size=3))
            if use_attention:
                self.decoder_attentions.append(CBAM(out_channels))
            in_channels = out_channels
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=1)
        )
    
    def forward(self, x):
        """
        x: (bs, seq_len, input_dim)
        输出: (bs, seq_len, output_dim)
        """
        bs, seq_len, dim = x.shape
        
        # 转换为 (bs, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 初始投影
        x = self.input_proj(x)  # (bs, hidden_dim, seq_len)
        
        # Encoder
        skip_connections = []
        for i in range(self.num_downsample):
            x = self.encoder_blocks[i](x)
            if self.use_attention:
                x = self.encoder_attentions[i](x)
            skip_connections.append(x)
            x = self.downsample[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_attention:
            x = self.bottleneck_attention(x)
        
        # Decoder
        for i in range(self.num_downsample):
            x = self.upsample[i](x)
            # 融合跳跃连接
            skip = skip_connections[self.num_downsample - 1 - i]
            x = torch.cat([x, skip], dim=1)
            x = self.fusion_layers[i](x)
            
            x = self.decoder_blocks[i](x)
            if self.use_attention:
                x = self.decoder_attentions[i](x)
        
        # 输出投影
        x = self.output_proj(x)  # (bs, output_dim, seq_len)
        
        # 转换回 (bs, seq_len, output_dim)
        x = x.transpose(1, 2)
        
        return x


class DynamicGatingNetwork(nn.Module):
    """
    改进的动态门控网络
    
    改进点:
    1. 使用LayerNorm替代BatchNorm - 更稳定
    2. 多头输出 - 更表达性强
    3. 时间步和噪声的独立处理路径
    """
    def __init__(self, hidden_dim=64, embed_dim=64, dtype=torch.bfloat16):
        super().__init__()
        
        self.time_proj = Timesteps(64, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedding = TimestepEmbedding(64, embed_dim)
        
        # 三个独立的处理路径
        self.thermal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.noise_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.time_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 多头门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dtype = dtype
    
    def forward(self, thermal_features, noise_latent, timestep):
        """
        thermal_features: (bs, seq_len, hidden_dim)
        noise_latent: (bs, seq_len, hidden_dim)
        timestep: (bs,)
        
        返回: (bs, seq_len, 1)
        """
        bs, seq_len, hidden_dim = thermal_features.shape
        
        # 时间步embedding
        time_emb = self.time_proj(timestep)
        time_emb = time_emb.to(self.dtype)
        time_emb = self.timestep_embedding(time_emb)  # (bs, embed_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, embed_dim)
        
        # 三个独立路径
        thermal_emb = self.thermal_proj(thermal_features)  # (bs, seq_len, hidden_dim)
        noise_emb = self.noise_proj(noise_latent)  # (bs, seq_len, hidden_dim)
        time_emb = self.time_fc(time_emb)  # (bs, seq_len, hidden_dim)
        
        # 拼接并生成门控权重
        fused = torch.cat([thermal_emb, noise_emb, time_emb], dim=-1)  # (bs, seq_len, hidden_dim*3)
        weight = self.gate(fused)  # (bs, seq_len, 1)
        weight = torch.sigmoid(weight)
        
        return weight


class MoGLE(nn.Module):
    """
    改进的MoGLE - 热成像到可见光转换
    
    改进点:
    1. UNet + 残差连接 + 注意力机制
    2. 改进的动态门控
    3. 灵活的配置选项
    4. 更好的数值稳定性
    """
    def __init__(
        self,
        input_dim=64,
        hidden_dim=256,
        output_dim=64,
        num_downsample=2,
        has_expert=True,
        has_gating=True,
        use_attention=True,
        use_residual=True,
        weight_is_scale=False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_expert = has_expert
        self.use_residual = use_residual
        
        # Expert网络
        if has_expert:
            self.expert = UNetExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_downsample=num_downsample,
                use_attention=use_attention
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
        thermal_latent: (bs, seq_len, input_dim)
        noise_latent: (bs, seq_len, input_dim)
        timestep: (bs,)
        
        返回: (bs, seq_len, output_dim)
        """
        # Expert处理
        expert_output = self.expert(thermal_latent)  # (bs, seq_len, output_dim)
        
        # 残差连接 - 如果input和output维度相同
        if self.use_residual and thermal_latent.shape == expert_output.shape:
            expert_output = expert_output + thermal_latent
        
        if self.gating is None:
            return expert_output
        
        # 动态门控
        weights = self.gating.forward(
            thermal_latent,
            noise_latent=noise_latent,
            timestep=timestep
        )  # (bs, seq_len, 1)
        
        # 加权融合
        output = expert_output * weights
        
        return output