import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.0, 1.0, 1000)
s_param = 0.008  # 余弦调度偏移
alpha_bar = np.cos(((t + s_param) / (1.0 + s_param)) * np.pi / 2.0) ** 2
# 数值尾部可能出现非常小的负数，做个截断
alpha_bar = np.clip(alpha_bar, 1e-12, 1.0)

s = np.sqrt(alpha_bar)
sigma = np.sqrt(1.0 - alpha_bar)

plt.figure(figsize=(6, 4))
plt.plot(t, s, label=r"$s(t)=\sqrt{\bar\alpha_t}$")
plt.plot(t, sigma, label=r"$\sigma(t)=\sqrt{1-\bar\alpha_t}$")
plt.title("VP (Cosine) — Signal/Noise Coefficients")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
# plt.savefig("vp_cosine_s_sigma.png", dpi=160)
plt.show()
