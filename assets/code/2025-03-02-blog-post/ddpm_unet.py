import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 正弦位置编码（时间步嵌入）
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# 残差块（包含时间步嵌入）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        # 时间步嵌入的MLP
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # 快捷连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        """
        x: (B, C, H, W) 输入特征
        t: (B, time_emb_dim) 时间步嵌入
        """
        # 第一个卷积块
        h = self.conv1(x)
        
        # 添加时间步信息
        time_emb = self.time_mlp(t)  # (B, out_channels)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        h = h + time_emb
        
        # 第二个卷积块
        h = self.conv2(h)
        
        # 快捷连接
        return h + self.shortcut(x)

# 下采样模块
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

# 上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        # 使用最近邻插值上采样
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

# DDPM UNet模型
class DDPMUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        time_emb_dim=256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 计算不同分辨率的通道数
        ch_mult = [1] + list(channel_mult)
        in_ch_mult = ch_mult[:-1]
        out_ch_mult = ch_mult[1:]
        
        # 下采样模块
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        
        # 存储下采样过程中的特征图（用于跳跃连接）
        self.down_features = []
        
        # 构建下采样路径
        for i, (in_mult, out_mult) in enumerate(zip(in_ch_mult, out_ch_mult)):
            out_channels = model_channels * out_mult
            
            # 添加残差块
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(current_channels, out_channels, time_emb_dim)
                )
                current_channels = out_channels
            
            # 如果不是最后一个块，添加下采样
            if i != len(in_ch_mult) - 1:
                self.down_blocks.append(Downsample(current_channels, current_channels))
        
        # 中间块
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim)
        
        # 上采样模块
        self.up_blocks = nn.ModuleList()
        
        # 构建上采样路径（反向顺序）
        for i, (in_mult, out_mult) in enumerate(zip(reversed(in_ch_mult), reversed(out_ch_mult))):
            out_channels = model_channels * out_mult
            
            # 添加上采样
            self.up_blocks.append(Upsample(current_channels, current_channels))
            
            # 添加残差块
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(current_channels + out_channels, out_channels, time_emb_dim)
                )
                current_channels = out_channels
        
        # 最终输出层
        self.final_norm = nn.GroupNorm(32, current_channels)
        self.final_conv = nn.Conv2d(current_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        """
        x: (B, C, H, W) 输入图像（带噪声）
        t: (B,) 时间步
        """
        # 时间步嵌入
        t_emb = self.time_embed(t)  # (B, time_emb_dim)
        
        # 初始卷积
        h = self.init_conv(x)
        
        # 存储下采样过程中的特征图（用于跳跃连接）
        features = []
        
        # 下采样路径
        for layer in self.down_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                features.append(h)
            else:
                h = layer(h)
        
        # 中间块
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        # 上采样路径
        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                # 从跳跃连接获取特征
                h_skip = features.pop()
                h = torch.cat([h, h_skip], dim=1)
                h = layer(h, t_emb)
        
        # 最终输出
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        return h

# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = DDPMUNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8)
    )
    
    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        output = model(x, t)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters())}")