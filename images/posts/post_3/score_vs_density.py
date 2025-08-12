import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 创建一个二维混合高斯分布作为目标密度
mean1 = np.array([1.5, 1.5])
cov1 = np.array([[0.1, 0], [0, 0.1]])
mean2 = np.array([-1.5, -1.5])
cov2 = np.array([[0.2, 0], [0, 0.2]])

rv1 = multivariate_normal(mean1, cov1)
rv2 = multivariate_normal(mean2, cov2)

# 网格
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 密度函数
p = rv1.pdf(pos) + rv2.pdf(pos)

# 计算 score function ∇ log p(x)
# 避免 log(0)，加一个小的 epsilon
eps = 1e-8
grad_x = (-(X - mean1[0]) / cov1[0, 0]) * rv1.pdf(pos) \
         + (-(X - mean2[0]) / cov2[0, 0]) * rv2.pdf(pos)
grad_y = (-(Y - mean1[1]) / cov1[1, 1]) * rv1.pdf(pos) \
         + (-(Y - mean2[1]) / cov2[1, 1]) * rv2.pdf(pos)

# ∇ log p = (∇p) / p
score_x = grad_x / (p + eps)
score_y = grad_y / (p + eps)

# 绘制密度背景和 score 向量场
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# 背景密度（颜色深浅）
ax.contourf(X, Y, p, levels=50, cmap='OrRd')

# score 向量场
ax.quiver(X, Y, score_x, score_y, color='black', scale=15)

ax.set_title("Density (background) and Score Function (vector field)", fontsize=12, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("density_score_vector_field.png")
plt.show()
