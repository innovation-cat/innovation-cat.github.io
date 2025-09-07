import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# ==============================================================================
#                      辅助模块 (Helper Modules)
# ==============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """将时间步 t 编码为正弦位置嵌入。"""
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

class Attention(nn.Module):
    """自注意力模块。"""
    def __init__(self, in_channels, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        hidden_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(8, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x, time_emb=None, class_emb=None):
        # 注意：为了统一接口，接收 time_emb 和 class_emb 但不使用它们
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.num_heads), qkv
        )
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        return self.to_out(out) + res

class ResBlock(nn.Module):
    """ADM 核心的残差块，包含 AdaGN 调制。"""
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, num_groups=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
            
        self.cond_emb_dim = time_emb_dim + class_emb_dim
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_emb_dim, out_channels * 2)
        )

    def forward(self, x, time_emb, class_emb):
        res = x
        cond_emb = torch.cat([time_emb, class_emb], dim=-1)
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        scale, shift = self.cond_mlp(cond_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.res_conv(res)

class ResBlockWithDownsample(nn.Module):
    """ADM 改进: BigGAN 风格的残差下采样块。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.GroupNorm(8, in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.skip_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x, time_emb=None, class_emb=None):
        # 注意：为了统一接口，接收 time_emb 和 class_emb 但不使用它们
        return self.main_path(x) + self.skip_path(x)

class ResBlockWithUpsample(nn.Module):
    """ADM 改进: BigGAN 风格的残差上采样块。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.GroupNorm(8, in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.skip_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x, time_emb=None, class_emb=None):
        # 注意：为了统一接口，接收 time_emb 和 class_emb 但不使用它们
        return self.main_path(x) + self.skip_path(x)


# ==============================================================================
#                        主 ADM UNET 模型 (完整修正版)
# ==============================================================================

class ADM_UNet_Complete(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=3,
        out_channels=3,
        model_channels=192,
        channel_mults=(1, 2, 3, 4),
        num_res_blocks=3,
        attention_resolutions=(8, 16, 32),
        num_classes=1001,
        dropout=0.1,
    ):
        super().__init__()

        self.model_channels = model_channels
        time_emb_dim = model_channels * 4

        # --- 1. 时间和类别嵌入 ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # --- 2. 初始卷积层 ---
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # --- 3. 下采样路径 (编码器) ---
        self.down_blocks = nn.ModuleList()
        current_res, ch, channels = image_size, model_channels, [model_channels]
        for i, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_emb_dim, time_emb_dim, dropout=dropout))
                ch = out_ch
                if current_res in attention_resolutions:
                    self.down_blocks.append(Attention(ch))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(ResBlockWithDownsample(ch, ch))
                current_res //= 2
                channels.append(ch)

        # --- 4. 网络瓶颈 (中间层) ---
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, time_emb_dim, dropout=dropout)
        self.mid_attn = Attention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, time_emb_dim, dropout=dropout)

        # --- 5. 上采样路径 (解码器) ---
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                in_ch = channels.pop() + ch
                self.up_blocks.append(ResBlock(in_ch, out_ch, time_emb_dim, time_emb_dim, dropout=dropout))
                ch = out_ch
                if current_res in attention_resolutions:
                    self.up_blocks.append(Attention(ch))
            if i != 0:
                self.up_blocks.append(ResBlockWithUpsample(ch, ch))
                current_res *= 2
        
        # --- 6. 最终输出层 ---
        self.final_norm = nn.GroupNorm(8, model_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, x, time, classes):
        t_emb = self.time_mlp(time)
        c_emb = self.class_emb(classes)
        
        x = self.init_conv(x)
        skip_connections = [x]

        # 下采样
        for block in self.down_blocks:
            x = block(x, t_emb, c_emb)
            skip_connections.append(x)
            
        # 瓶颈
        x = self.mid_block1(x, t_emb, c_emb)
        x = self.mid_attn(x, t_emb, c_emb)
        x = self.mid_block2(x, t_emb, c_emb)
        
        # 上采样
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb, c_emb)
            
        # 输出
        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)


# ==============================================================================
#                                  测试代码
# ==============================================================================
if __name__ == '__main__':
    # 模拟输入
    batch_size = 2
    image_size = 64
    input_channels = 3
    
    noisy_images = torch.randn(batch_size, input_channels, image_size, image_size)
    timesteps = torch.randint(0, 1000, (batch_size,)).long()
    classes = torch.randint(0, 1001, (batch_size,)).long()

    # 创建 ADM UNET 模型
    model = ADM_UNet_Complete(
        image_size=image_size,
        in_channels=input_channels,
        out_channels=input_channels,
        model_channels=128,          
        channel_mults=(1, 2, 2, 2),  
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        num_classes=1001
    )

    # 前向传播
    predicted_noise = model(noisy_images, timesteps, classes)

    # 检查输出形状
    print(f"输入图像形状: {noisy_images.shape}")
    print(f"模型输出（预测噪声）形状: {predicted_noise.shape}")
    assert noisy_images.shape == predicted_noise.shape, "输入和输出形状不匹配！"
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {num_params / 1e6:.2f} M")