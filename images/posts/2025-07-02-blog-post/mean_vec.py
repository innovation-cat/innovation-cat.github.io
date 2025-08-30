import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Load a specific Chinese font
font = FontProperties(family='SimHei', size=12)  # Or 'PingFang SC', 'Noto Sans CJK SC', etc.
# 设置随机种子
np.random.seed(0)

# 随机生成一些pair (x0, x1)
num_pairs = 30
x0 = np.random.randn(num_pairs, 2) * 0.5 + np.array([0, -2])  # 起点（噪声分布）
x1 = np.random.randn(num_pairs, 2) * 0.5 + np.array([0, 2])   # 终点（数据分布）

# 绘制pair的直线路径
t_vals = np.linspace(0, 1, 20)
for i in range(num_pairs):
    xs = (1 - t_vals)[:, None] * x0[i] + t_vals[:, None] * x1[i]
    plt.plot(xs[:, 0], xs[:, 1], 'gray', alpha=0.6)

# 构造一个弯曲的“平均速度场”，模拟条件期望效果
# 用一个旋转场 + 向上的偏移
xgrid, ygrid = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2.5, 2.5, 20))
u = -0.3 * ygrid
v = 1.0 + 0.3 * xgrid

plt.quiver(xgrid, ygrid, u, v, color="blue", alpha=0.5)

# 模拟一条采样轨迹（在平均场中积分出来的曲线）
traj_t = np.linspace(0, 10, 80)
x_traj = np.zeros((len(traj_t), 2))
x_traj[0] = np.array([0, -2])
dt = 0.05
for i in range(1, len(traj_t)):
    x, y = x_traj[i-1]
    dx = -0.3 * y
    dy = 1.0 + 0.3 * x
    x_traj[i] = x_traj[i-1] + dt * np.array([dx, dy])

plt.plot(x_traj[:, 0], x_traj[:, 1], 'r-', linewidth=2, label="采样轨迹")

# 标注
plt.scatter(x0[:, 0], x0[:, 1], c='black', marker='x', label="起点 x0")
plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='o', label="终点 x1")

plt.legend()
plt.title("直线路径 vs 平均速度场采样轨迹")
plt.axis("equal")
plt.show()
