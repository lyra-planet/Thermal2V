import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.nn import functional as F


class ResidualConvBlock(nn.Module):
    """残差卷积块"""
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
    """通道注意力"""
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
    """空间注意力 - 关键改进：识别结构关键点"""
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


class StructuralAwareAttention(nn.Module):
    """
    结构感知注意力 - 关键创新点
    专门为了捕捉人脸结构信息（角度、表情、姿态等）
    """
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        
        # 多尺度卷积核 - 捕捉不同粒度的结构
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(channels, channels // len(kernel_sizes), k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # 梯度计算 - 捕捉变化率（结构边界）
        self.gradient_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
        # 局部一致性 - 捕捉连贯结构
        self.coherence_fc = nn.Sequential(
            nn.Linear(channels * (1 + len(kernel_sizes)), channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (bs, channels, seq_len)
        """
        bs, c, seq_len = x.shape
        
        # 1. 多尺度特征
        multi_scale_feats = []
        for conv in self.multi_scale_convs:
            feat = conv(x)
            multi_scale_feats.append(feat)
        multi_scale = torch.cat(multi_scale_feats, dim=1)  # (bs, c, seq_len)
        
        # 2. 梯度特征 - 捕捉结构边界和变化
        grad_feat = self.gradient_conv(x)
        grad_mag = torch.abs(grad_feat)  # 梯度幅度
        
        # 3. 特征融合和局部一致性
        combined = torch.cat([x, multi_scale], dim=1)  # (bs, c*(1+len(ks)), seq_len)
        combined = combined.transpose(1, 2)  # (bs, seq_len, c*(1+len(ks)))
        combined_flat = combined.reshape(bs * seq_len, -1)
        
        # 计算注意力权重
        attention_weights = self.coherence_fc(combined_flat)  # (bs*seq_len, c)
        attention_weights = attention_weights.reshape(bs, seq_len, c).transpose(1, 2)  # (bs, c, seq_len)
        
        # 融合多尺度和梯度信息
        output = x * attention_weights + grad_mag * 0.3
        
        return output


class CBSAMPlus(nn.Module):
    """
    增强的CBAM - 融合结构感知能力
    """
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.structural_attention = StructuralAwareAttention(channels)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.structural_attention(x)
        return x


class StructurePreservingUNetExpert(nn.Module):
    """
    结构保留的UNet Expert
    
    关键改进点用于捕捉人物结构信息:
    1. 结构感知注意力 - 重点关注结构变化
    2. 多层次梯度约束 - 保留细节变化
    3. 特征分离 - 分别处理几何和外观特征
    4. 结构约束解码器 - 上采样时保留结构
    """
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=64, num_downsample=2, use_structure_attention=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_downsample = num_downsample
        self.use_structure_attention = use_structure_attention
        
        # 初始投影
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # 几何特征路径 - 捕捉结构信息（人脸角度、姿态等）
        self.geometry_encoder = self._build_geometry_encoder(hidden_dim)
        
        # 外观特征路径 - 捕捉颜色和纹理信息
        self.appearance_encoder = self._build_appearance_encoder(hidden_dim)
        
        # 融合层 - 修复：输入通道数应该是 hidden_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Encoder (下采样路径)
        self.encoder_blocks = nn.ModuleList()
        self.structure_attentions = nn.ModuleList() if use_structure_attention else None
        self.downsample = nn.ModuleList()
        
        in_channels = hidden_dim
        for i in range(num_downsample):
            self.encoder_blocks.append(ResidualConvBlock(in_channels, kernel_size=3))
            if use_structure_attention:
                self.structure_attentions.append(CBSAMPlus(in_channels))
            self.downsample.append(nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1))
            in_channels *= 2
        
        # Bottleneck
        self.bottleneck = ResidualConvBlock(in_channels, kernel_size=3)
        if use_structure_attention:
            self.bottleneck_attention = CBSAMPlus(in_channels)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_structure_attentions = nn.ModuleList() if use_structure_attention else None
        self.upsample = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i in range(num_downsample):
            out_channels = in_channels // 2
            self.upsample.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            self.fusion_layers.append(nn.Sequential(
                nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            ))
            self.decoder_blocks.append(ResidualConvBlock(out_channels, kernel_size=3))
            if use_structure_attention:
                self.decoder_structure_attentions.append(CBSAMPlus(out_channels))
            in_channels = out_channels
        
        # 输出层 - 保留结构信息
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=1)
        )
        
        # 结构引导损失的辅助输出头（训练时可用）
        self.structure_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 16, kernel_size=1)  # 预测结构关键点
        )
    
    def _build_geometry_encoder(self, hidden_dim):
        """几何编码器 - 关注结构信息"""
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )
    
    def _build_appearance_encoder(self, hidden_dim):
        """外观编码器 - 关注颜色纹理"""
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )
    
    def forward(self, x, return_structure_logits=False):
        """
        x: (bs, seq_len, input_dim)
        return_structure_logits: 训练时返回结构辅助输出
        """
        bs, seq_len, dim = x.shape
        x = x.transpose(1, 2)  # (bs, input_dim, seq_len)
        
        # 初始投影
        x = self.input_proj(x)  # (bs, hidden_dim, seq_len)
        
        # 双路径编码
        geometry_feat = self.geometry_encoder(x)  # (bs, hidden_dim, seq_len)
        appearance_feat = self.appearance_encoder(x)  # (bs, hidden_dim, seq_len)
        x = self.fusion_layer(torch.cat([geometry_feat, appearance_feat], dim=1))
        
        # Encoder
        skip_connections = []
        for i in range(self.num_downsample):
            x = self.encoder_blocks[i](x)
            if self.use_structure_attention:
                x = self.structure_attentions[i](x)
            skip_connections.append(x)
            x = self.downsample[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_structure_attention:
            x = self.bottleneck_attention(x)
        
        # 结构辅助输出（训练时用）
        structure_logits = None
        if return_structure_logits:
            structure_logits = self.structure_head(x)
        
        # Decoder
        for i in range(self.num_downsample):
            x = self.upsample[i](x)
            skip = skip_connections[self.num_downsample - 1 - i]
            x = torch.cat([x, skip], dim=1)
            x = self.fusion_layers[i](x)
            
            x = self.decoder_blocks[i](x)
            if self.use_structure_attention:
                x = self.decoder_structure_attentions[i](x)
        
        # 输出投影
        x = self.output_proj(x)
        x = x.transpose(1, 2)  # (bs, seq_len, output_dim)
        
        if return_structure_logits:
            return x, structure_logits
        return x


class DynamicGatingNetwork(nn.Module):
    """改进的动态门控网络"""
    def __init__(self, hidden_dim=64, embed_dim=64, dtype=torch.bfloat16):
        super().__init__()
        
        self.time_proj = Timesteps(64, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedding = TimestepEmbedding(64, embed_dim)
        
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
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dtype = dtype
    
    def forward(self, thermal_features, noise_latent, timestep):
        bs, seq_len, hidden_dim = thermal_features.shape
        
        time_emb = self.time_proj(timestep)
        time_emb = self.timestep_embedding(time_emb)
        time_emb = time_emb.to(thermal_features.dtype)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        thermal_emb = self.thermal_proj(thermal_features)
        noise_emb = self.noise_proj(noise_latent)
        time_emb = self.time_fc(time_emb)
        
        fused = torch.cat([thermal_emb, noise_emb, time_emb], dim=-1)
        weight = self.gate(fused)
        weight = torch.sigmoid(weight)
        
        return weight


class MoGLE(nn.Module):
    """
    MoGLE - 增强版本，重点优化结构信息捕捉
    
    关键特性:
    1. 结构感知注意力 - 专门捕捉人脸角度、表情等
    2. 双路径编码 - 分离几何和外观信息
    3. 结构约束 - 通过辅助任务引导学习
    4. 梯度约束 - 保留细节变化
    """
    def __init__(
        self,
        input_dim=64,
        hidden_dim=256,
        output_dim=64,
        num_downsample=2,
        has_expert=True,
        has_gating=True,
        use_structure_attention=True,
        weight_is_scale=False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_expert = has_expert
        
        if has_expert:
            self.expert = StructurePreservingUNetExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_downsample=num_downsample,
                use_structure_attention=use_structure_attention
            )
        else:
            self.expert = nn.Identity()
        
        if has_gating:
            self.gating = DynamicGatingNetwork(hidden_dim=input_dim)
        else:
            self.gating = None
        self.weight_is_scale = weight_is_scale
    
    def forward(self, thermal_latent, noise_latent, timestep, return_structure_logits=False):
        """
        thermal_latent: (bs, seq_len, input_dim)
        noise_latent: (bs, seq_len, input_dim)
        timestep: (bs,)
        return_structure_logits: 是否返回结构辅助输出
        
        返回: 
        - 如果return_structure_logits=True: (output, structure_logits)
        - 否则: output (bs, seq_len, output_dim)
        """
        if return_structure_logits and isinstance(self.expert, StructurePreservingUNetExpert):
            expert_output, structure_logits = self.expert(thermal_latent, return_structure_logits=True)
        else:
            expert_output = self.expert(thermal_latent)
            structure_logits = None
        
        if self.gating is None:
            if return_structure_logits:
                return expert_output, structure_logits
            return expert_output
        
        weights = self.gating.forward(
            thermal_latent,
            noise_latent=noise_latent,
            timestep=timestep
        )
        
        output = expert_output * weights
        
        if return_structure_logits:
            return output, structure_logits
        return output