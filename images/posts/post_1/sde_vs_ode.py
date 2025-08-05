import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from matplotlib.animation import PillowWriter  # 需要保存 GIF 时解注

# ========= 1D toy reverse dynamics =========
# score(x,t) = ∇ log p_t(x) ≈ -k(t) x
def k(t): return 2.0 + 1.0 * t
def beta(t): return 4.0
def f(t): return -0.5 * beta(t)
def g(t): return np.sqrt(beta(t))

# SDE: dx = [ f(t) + g(t)^2 * k(t) ] x dt + g(t) dW
def drift_sde(t, x): return (f(t) + g(t)**2 * k(t)) * x
# PF-ODE: dx = [ f(t) + 0.5 g(t)^2 * k(t) ] x dt
def drift_ode(t, x): return (f(t) + 0.5 * g(t)**2 * k(t)) * x

# ========= 时间网格：从右到左（1 → 0） =========
N = 100
t_rev = np.linspace(1.0, 0.0, N)   # 用于计算和绘图（递减）
dt = t_rev[1] - t_rev[0]           # 负数
h = abs(dt)

# ========= 初值 =========
x0 = 2.0

# ========= 模拟大量 SDE 轨迹（只取统计量） =========
num_paths = 3000
rng = np.random.default_rng(0)

X = np.full((num_paths,), x0, dtype=float)
X_all = np.empty((N, num_paths), dtype=float)
X_all[0] = X
for j, t in enumerate(t_rev[:-1]):
    dW = np.sqrt(h) * rng.normal(size=num_paths)
    X = X + drift_sde(t, X) * dt + g(t) * dW
    X_all[j + 1] = X

sde_mean = X_all.mean(axis=1)
q05 = np.quantile(X_all, 0.05, axis=1)
q25 = np.quantile(X_all, 0.25, axis=1)
q50 = np.quantile(X_all, 0.50, axis=1)
q75 = np.quantile(X_all, 0.75, axis=1)
q95 = np.quantile(X_all, 0.95, axis=1)

# ========= 单条 ODE 轨迹（同样在 t_rev 上积分） =========
x = x0
xs_ode = np.empty(N, dtype=float); xs_ode[0] = x
for j, t in enumerate(t_rev[:-1]):
    x = x + drift_ode(t, x) * dt
    xs_ode[j + 1] = x

# ========= 画布与元素 =========
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.set_xlim(1.0, 0.0)  # 右端=1，左端=0（方向反转）
# 先用两个点初始化，避免 fill_between 单点退化
k0 = 2
band95 = ax.fill_between(t_rev[:k0], q05[:k0], q95[:k0], alpha=0.18, label="SDE 5–95% band")
band75 = ax.fill_between(t_rev[:k0], q25[:k0], q75[:k0], alpha=0.28, label="SDE 25–75% band")

line_mean, = ax.plot(t_rev[:k0], sde_mean[:k0], linestyle="--", linewidth=1.8, label="SDE mean")
# 若需中位数：line_med, = ax.plot(t_rev[:k0], q50[:k0], linestyle=":", linewidth=1.4, label="SDE median")

line_ode,  = ax.plot(t_rev[:k0], xs_ode[:k0], linewidth=2.6, label="PF-ODE path")
pt_mean,   = ax.plot([t_rev[k0-1]], [sde_mean[k0-1]], marker="o", ms=5)
pt_ode,    = ax.plot([t_rev[k0-1]], [xs_ode[k0-1]],  marker="o", ms=6)

ax.set_title("Reverse SDE (quantile bands + mean) vs Probability-flow ODE (single path)")
ax.set_xlabel("t (left: 1 → right: 0)")
ax.set_ylabel("x(t)")
ax.grid(True, alpha=0.25)
ax.legend(loc="best")

pad = 0.05 * (np.max(q95) - np.min(q05) + 1e-8)
ax.set_ylim(np.min(q05) - pad, np.max(q95) + pad)

txt = ax.text(0.02, 0.95, f"t = {t_rev[k0-1]:.3f}", transform=ax.transAxes, va="top")

def update(frame):
    global band95, band75
    k = max(2, frame)  # 每帧使用 t_rev[:k]，从右→左逐步扩展
    t_seg = t_rev[:k]

    # 重绘分位带
    for coll in [band95, band75]:
        try:
            coll.remove()
        except Exception:
            pass
    band95 = ax.fill_between(t_seg, q05[:k], q95[:k], alpha=0.18)
    band75 = ax.fill_between(t_seg, q25[:k], q75[:k], alpha=0.28)

    # 更新曲线与当前点
    line_mean.set_data(t_seg, sde_mean[:k])
    # line_med.set_data(t_seg, q50[:k])  # 若启用中位数
    line_ode.set_data(t_seg, xs_ode[:k])

    pt_mean.set_data([t_seg[-1]], [sde_mean[k-1]])
    pt_ode.set_data([t_seg[-1]],  [xs_ode[k-1]])

    txt.set_text(f"t = {t_seg[-1]:.3f}")
    return band95, band75, line_mean, line_ode, pt_mean, pt_ode, txt

anim = FuncAnimation(fig, update, frames=N, interval=20, blit=False)
plt.tight_layout()
plt.show()

# ========== 保存（2 选 1） ==========
# 1) GIF（不依赖 ffmpeg）
# anim.save("sde_vs_ode_1d_right_to_left_fixed.gif", writer=PillowWriter(fps=24))
# 2) MP4（需要 ffmpeg）
# anim.save("sde_vs_ode_1d_right_to_left_fixed.mp4", fps=30, dpi=160)


# ========== 保存（2 选 1） ==========
# 1) GIF（不依赖 ffmpeg）
# anim.save("sde_vs_ode_1d_right_to_left.gif", writer=PillowWriter(fps=24))

# 2) MP4（需要系统安装 ffmpeg）
# anim.save("sde_vs_ode_1d_right_to_left.mp4", fps=30, dpi=160)


# ========== 保存（2 选 1） ==========
# 保存 GIF（不依赖 ffmpeg）
anim.save("sde_vs_ode_1d.gif", writer='pillow', fps=15, dpi=100)
#ani.save('flow_matching.gif', writer='pillow', fps=30)

# 保存 MP4（需要系统安装 ffmpeg）
# anim.save("sde_vs_ode_1d.mp4", fps=30, dpi=160)
