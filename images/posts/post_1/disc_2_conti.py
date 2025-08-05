import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.axis("off")

# 左：离散
plt.text(
    0.15, 0.70,
    r"Discrete schedule" + "\n" + r"$x_t=s(t)\,x_0+\sigma(t)\,\varepsilon$",
    fontsize=12, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
)

# 右：连续 SDE
plt.text(
    0.65, 0.70,
    r"Continuous SDE" + "\n" + r"$dx_t=f(t)\,x_t\,dt+g(t)\,dw_t$",
    fontsize=12, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
)

# 左->右（公式）
plt.annotate("", xy=(0.58, 0.73), xytext=(0.36, 0.73), arrowprops=dict(arrowstyle="->"))
plt.text(0.47, 0.78, r"$f(t)=\frac{s'(t)}{s(t)}$", ha="center", fontsize=11)
plt.text(0.47, 0.72, r"$g^2(t)=2\!\left(\sigma\sigma'-\frac{s'}{s}\sigma^2\right)$", ha="center", fontsize=11)

# 右->左（逆映射）
plt.annotate("", xy=(0.36, 0.62), xytext=(0.58, 0.62), arrowprops=dict(arrowstyle="->"))
plt.text(0.47, 0.56, r"$s(t)=\phi(t)=\exp\!\left(\int_0^t f(s)\,ds\right)$", ha="center", fontsize=11)
plt.text(0.47, 0.50, r"$\sigma^2(t)=\phi^2\!\int_0^t \frac{g^2}{\phi^2}\,ds$", ha="center", fontsize=11)

plt.title("(s, σ)  ↔  (f, g)", pad=10)
plt.tight_layout()
# plt.savefig("mapping_s_sigma_to_f_g.png", dpi=160)
plt.show()
