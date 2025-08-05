import numpy as np
import matplotlib.pyplot as plt

# 时间归一化到 [0,1]
t = np.linspace(0.0, 1.0, 1000)

# 线性 beta 调度（示例数值，可根据你的文章调整）
beta0, beta1 = 1e-4, 0.02
# ∫0..t beta(s) ds = beta0 * t + 0.5 * (beta1 - beta0) * t^2
int_beta = 1000*(beta0 * t + 0.5 * (beta1 - beta0) * t**2)

alpha_bar = np.exp(-int_beta)
s = np.sqrt(alpha_bar)
print(s[0:1000:30])
sigma = np.sqrt(1.0 - alpha_bar)

plt.figure(figsize=(6, 4))
plt.plot(t, s, label=r"$s(t)=\sqrt{\bar\alpha_t}$")
plt.plot(t, sigma, label=r"$\sigma(t)=\sqrt{1-\bar\alpha_t}$")
plt.title("VP (Linear) — Signal/Noise Coefficients")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
# 可选：保存
plt.savefig("vp_linear_s_sigma.png", dpi=160)
plt.show()
