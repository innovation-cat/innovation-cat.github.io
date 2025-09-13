import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 核心互信息函数 (基于SNR)
# -----------------------------------------------------------------------------
def mi_x0_pred(snr):
    """Mutual Information for x0-prediction vs. SNR (based on Gaussian assumption)."""
    # I = 0.5 * log(1 + SNR)
    return 0.5 * np.log(1 + snr)

def mi_eps_pred(snr):
    """Mutual Information for epsilon-prediction vs. SNR (based on Gaussian assumption)."""
    # I = 0.5 * log(1 + 1/SNR)
    # Add a small epsilon to avoid log(inf) for snr=0
    return 0.5 * np.log(1 + 1 / (snr + 1e-9))

def mi_v_pred_illustrative(snr):
    """Illustrative arch-shaped Mutual Information for v-prediction."""
    # This is a qualitative curve to show the arch-shape for non-Gaussian data.
    log_snr = np.log(snr + 1e-9)
    peak_location = 0  # Peak at log(SNR)=0, i.e., SNR=1
    width = 1.8
    amplitude = 1.0 
    return amplitude * np.exp(-(log_snr - peak_location)**2 / (2 * width**2))

# -----------------------------------------------------------------------------
# 2. 从时间步 t 到 SNR 的映射 (使用余弦噪声计划)
# -----------------------------------------------------------------------------
def t_to_snr_cosine(t, T=1000, s=0.008):
    """Maps timestep t to SNR based on a cosine schedule."""
    # Normalize t to be in [0, 1]
    t_normalized = t / T
    # Calculate cumulative alpha (alphas_cumprod) based on the cosine schedule formula
    f_t = np.cos((t_normalized + s) / (1 + s) * np.pi / 2)**2
    alphas_cumprod = f_t
    # SNR = alpha_cumprod / (1 - alpha_cumprod)
    # Add a small epsilon to avoid division by zero
    return alphas_cumprod / (1 - alphas_cumprod + 1e-9)

# -----------------------------------------------------------------------------
# 3. 数据生成
# -----------------------------------------------------------------------------
# --- Data for MI vs. SNR plot ---
snr_axis = np.logspace(-5, 5, 500)
mi_x0_snr = mi_x0_pred(snr_axis)
mi_eps_snr = mi_eps_pred(snr_axis)
mi_v_snr = mi_v_pred_illustrative(snr_axis)

# --- Data for MI vs. Timestep plot ---
T = 1000
t_axis = np.arange(0, T)
snr_from_t = t_to_snr_cosine(t_axis, T)

mi_x0_t = mi_x0_pred(snr_from_t)
mi_eps_t = mi_eps_pred(snr_from_t)
mi_v_t = mi_v_pred_illustrative(snr_from_t)

# -----------------------------------------------------------------------------
# 4. 绘图
# -----------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# --- Plot 1: MI vs. Timestep ---
ax1.plot(t_axis, mi_x0_t, label='$I(x_0; x_t)$', color='royalblue', linewidth=2.5)
ax1.plot(t_axis, mi_eps_t, label='$I(\\epsilon; x_t)$', color='seagreen', linewidth=2.5)
ax1.plot(t_axis, mi_v_t, label='$I(v; x_t)$ (illustrative)', color='firebrick', linestyle='--', linewidth=2.5)
ax1.set_xlabel('Timestep (t)', fontsize=14)
ax1.set_ylabel('Mutual Information (nats)', fontsize=14)
ax1.set_title('View 1: MI vs. Timestep\n(Cosine Schedule)', fontsize=16, pad=15)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlim(0, T)

# --- Plot 2: MI vs. SNR ---
ax2.plot(snr_axis, mi_x0_snr, label='$I(x_0; x_t)$ ($x_0$-pred)', color='royalblue', linewidth=2.5)
ax2.plot(snr_axis, mi_eps_snr, label='$I(\\epsilon; x_t)$ ($\\epsilon$-pred)', color='seagreen', linewidth=2.5)
ax2.plot(snr_axis, mi_v_snr, label='$I(v; x_t)$ ($v$-pred, illustrative)', color='firebrick', linestyle='--', linewidth=2.5)
ax2.set_xscale('log')
ax2.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=14)
ax2.set_ylabel('Mutual Information (nats)', fontsize=14)
ax2.set_title('View 2: MI vs. SNR\n(Fundamental View)', fontsize=16, pad=15)
ax2.legend(fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_xlim(1e-5, 1e5)

# --- Global Figure Settings ---
fig.suptitle('Mutual Information Dynamics for Different Prediction Targets', fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()