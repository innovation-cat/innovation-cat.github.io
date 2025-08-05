import numpy as np
import matplotlib.pyplot as plt

# ===== 时间与常量 =====
t = np.linspace(0.0, 1.0, 1000)

# ===== VP (Linear) =====
# 参照 DDPM：离散步数常取 T=1000, beta ∈ [1e-4, 0.02]
T = 1000
beta0, beta1 = 1e-4, 0.02
# ∫ beta(s) ds 的连续化，并乘以 T 使其与离散总量匹配
int_beta = T * (beta0 * t + 0.5 * (beta1 - beta0) * t**2)
alpha_bar_lin = np.exp(-int_beta)
s_lin = np.sqrt(alpha_bar_lin)
sigma_lin = np.sqrt(1.0 - alpha_bar_lin)

# ===== VP (Cosine) =====
# iDDPM 余弦：alpha_bar(t) ∝ cos^2((t+s)/(1+s)*π/2)
# 做归一化使得 alpha_bar(0)=1
s_param = 0.008
f = np.cos(((t + s_param) / (1.0 + s_param)) * np.pi / 2.0) ** 2
alpha_bar_cos = np.clip(f / f[0], 1e-12, 1.0)
s_cos = np.sqrt(alpha_bar_cos)
sigma_cos = np.sqrt(1.0 - alpha_bar_cos)

# ===== VE =====
# s(t)=1, σ(t)=σ_min*(σ_max/σ_min)^t
sigma_min, sigma_max = 0.01, 50.0
s_ve = np.ones_like(t)
sigma_ve = sigma_min * (sigma_max / sigma_min) ** t

# ===== 画图：同一画布、三列子图 =====
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)

# VP-linear
axes[0].plot(t, s_lin, label=r"$s(t)=\sqrt{\bar\alpha_t}$")
axes[0].plot(t, sigma_lin, label=r"$\sigma(t)=\sqrt{1-\bar\alpha_t}$")
axes[0].set_title("VP (Linear)")
axes[0].set_xlabel("t")
axes[0].set_ylabel("value")
axes[0].legend()
axes[0].grid(True, alpha=0.25)

# VP-cos
axes[1].plot(t, s_cos, label=r"$s(t)=\sqrt{\bar\alpha_t}$")
axes[1].plot(t, sigma_cos, label=r"$\sigma(t)=\sqrt{1-\bar\alpha_t}$")
axes[1].set_title("VP (Cosine)")
axes[1].set_xlabel("t")
axes[1].legend()
axes[1].grid(True, alpha=0.25)

# VE（对数纵轴）
axes[2].plot(t, s_ve, label=r"$s(t)=1$")
axes[2].plot(t, sigma_ve, label=r"$\sigma(t)$ (geometric)")
axes[2].set_yscale("log")
axes[2].set_title("VE (log scale)")
axes[2].set_xlabel("t")
axes[2].legend()
axes[2].grid(True, which="both", alpha=0.25)

plt.tight_layout()
plt.show()

# 如果希望 t 轴右端=1、左端=0（右→左显示），取消注释以下两行：
# for ax in axes:
#     ax.set_xlim(1.0, 0.0)
