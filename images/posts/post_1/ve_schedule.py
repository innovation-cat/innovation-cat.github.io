import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.0, 1.0, 1000)
sigma_min, sigma_max = 0.01, 50.0

s = np.ones_like(t)
sigma = sigma_min * (sigma_max / sigma_min) ** t  # 几何增长

plt.figure(figsize=(6, 4))
plt.plot(t, s, label=r"$s(t)=1$")
plt.plot(t, sigma, label=r"$\sigma(t)$ (geometric)")
plt.yscale("log")  # VE 的噪声跨越数量级，建议对数显示
plt.title("VE — Signal/Noise Coefficients")
plt.xlabel("t")
plt.ylabel("value (log scale)")
plt.legend()
plt.tight_layout()
# plt.savefig("ve_s_sigma.png", dpi=160)
plt.show()
