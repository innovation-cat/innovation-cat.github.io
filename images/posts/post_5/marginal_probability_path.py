import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 1. 参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 1000                 # 点的个数
T = 100                 # 时间步数
dt = 1.0 / T             # 每步时间长度

# 2. 采样初始点：标准正态，中心移到 (-8, 0)
x0 = torch.randn(N, 2, device=device) + torch.tensor([-8.0, 0.0], device=device)

# 3. 采样目标点：2D 三峰混合高斯
pis = torch.tensor([0.3, 0.4, 0.3], device=device)
mus = torch.tensor([
    [5.0,  0.0],
    [4.0,  3.0],
    [4.0, -3.0],
], device=device)
sigma = 0.7

cat = torch.distributions.Categorical(probs=pis)
comp_idx = cat.sample((N,))             # shape (N,)
eps = torch.randn(N, 2, device=device)  # shape (N,2)
y  = mus[comp_idx] + sigma * eps        # shape (N,2)

# 4. 生成等高线网格
x = np.linspace(-10, 7, 300)
y_grid = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(x, y_grid)

# 初始分布密度：标准正态，均值 (-8,0)
Z0 = (1 / (2 * np.pi)) * np.exp(-0.5 * ((X + 8)**2 + Y**2))

# 目标混合高斯密度
Z = np.zeros_like(X)
pis_np = pis.cpu().numpy()
mus_np = mus.cpu().numpy()
for k in range(len(pis_np)):
    mu_x, mu_y = mus_np[k]
    Z += pis_np[k] * (1 / (2 * np.pi * sigma**2)) * \
         np.exp(-((X - mu_x)**2 + (Y - mu_y)**2) / (2 * sigma**2))

# 5. 初始化绘图，背景设为白色
pos = x0.clone()
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(-10, 7)
ax.set_ylim(-5, 5)
ax.set_axis_off()  # 去掉所有坐标轴和刻度
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_title('Flow Matching: From N([-8,0],I) to 3-Modal Mixture', color='black')

# 用 contourf 绘制填充等高线
cf0 = ax.contourf(X, Y, Z0, levels=10, cmap='Blues', alpha=0.4)
cf1 = ax.contourf(X, Y, Z,  levels=10, cmap='Reds',  alpha=0.4)

# 绘制散点
scat = ax.scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), s=5, alpha=0.6, c='black')

# 6. 向量场定义
def vector_field(x, y, t):
    return (y - x) / (1.0 - t)

# 7. 初始化与更新函数
def init():
    scat.set_offsets(x0.cpu().numpy())
    return scat,

def animate(frame):
    global pos
    t = frame * dt
    v = vector_field(pos, y, t)
    pos = pos + v * dt
    scat.set_offsets(pos.cpu().numpy())
    return scat,

# 8. 创建动画
ani = FuncAnimation(
    fig, animate, init_func=init,
    frames=T, interval=20, blit=True
)

# 9. 保存为 GIF (需要安装 pillow: pip install pillow)
ani.save('flow_matching.gif', writer='pillow', fps=30)
#writer = PillowWriter(fps=30)
#ani.save('flow_matching.gif', writer=writer, savefig_kwargs={'bbox_inches':'tight', 'pad_inches':0})
plt.close(fig)

