# Re-draw SNR (dB) comparison for 5 beta-based schedules:
# linear, scaled_linear, cosine (squaredcos_cap_v2), sigmoid (diffusers-style), exponential (geometric β).
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----- Config -----
T = 1000
t = np.arange(T)
tau = t / (T - 1)

# Baselines for β ranges
beta_linear_start = 1e-4
beta_linear_end = 2e-2

beta_scaled_start = 0.00085
beta_scaled_end = 0.012

# ----- Helpers -----
def snr_from_betas(betas: np.ndarray) -> np.ndarray:
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    # SNR for VP/DDPM: alpha_bar / (1 - alpha_bar)
    eps = 1e-12
    snr = alpha_bar / np.maximum(1 - alpha_bar, eps)
    return snr, alpha_bar

# 1) Linear β-schedule
betas_linear = np.linspace(beta_linear_start, beta_linear_end, T)
snr_linear, abar_linear = snr_from_betas(betas_linear)

# 2) Scaled linear β-schedule
betas_scaled = np.linspace(beta_scaled_start, beta_scaled_end, T)
snr_scaled, abar_scaled = snr_from_betas(betas_scaled)

# 3) Cosine schedule via alpha_bar (squaredcos_cap_v2)
def alpha_bar_cosine(tau: np.ndarray, s: float = 0.008) -> np.ndarray:
    num = np.cos(((tau + s) / (1 + s)) * np.pi / 2.0) ** 2
    den = np.cos((s / (1 + s)) * np.pi / 2.0) ** 2
    return num / den

alpha_bar_cos = alpha_bar_cosine(tau, s=0.008)
alphas_cos = np.empty_like(alpha_bar_cos)
alphas_cos[0] = alpha_bar_cos[0]
alphas_cos[1:] = alpha_bar_cos[1:] / np.maximum(alpha_bar_cos[:-1], 1e-12)
betas_cos = 1.0 - alphas_cos
betas_cos = np.clip(betas_cos, 0.0, 0.999)  # mimic common clamp
snr_cos, abar_cos = snr_from_betas(betas_cos)

# 4) Sigmoid β-schedule (diffusers style) on β directly
z = -6 + 12 * tau
betas_sigmoid = 1.0 / (1.0 + np.exp(-z))
betas_sigmoid = betas_sigmoid * (beta_linear_end - beta_linear_start) + beta_linear_start
snr_sigmoid, abar_sigmoid = snr_from_betas(betas_sigmoid)

# 5) Exponential β-schedule (geometric progression between the same endpoints as linear)
betas_exp = beta_linear_start * (beta_linear_end / beta_linear_start) ** tau
snr_exp, abar_exp = snr_from_betas(betas_exp)



# ----- Plot -----
plt.figure(figsize=(9,6))
plt.plot(tau, 10*np.log10(snr_linear + 1e-12), label="linear")
plt.plot(tau, 10*np.log10(snr_scaled + 1e-12), label="scaled_linear")
plt.plot(tau, 10*np.log10(snr_cos + 1e-12), label="cosine")
plt.plot(tau, 10*np.log10(snr_sigmoid + 1e-12), label="sigmoid")
plt.plot(tau, 10*np.log10(snr_exp + 1e-12), label="exponential")
plt.xlabel("Normalized time t/T")
plt.ylabel("SNR (dB)")
plt.title("SNR comparison (β-based schedules): linear / scaled_linear / cosine / sigmoid / exponential")
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.legend()
png_path = "snr_beta5_schedules.png"
plt.tight_layout()
plt.savefig(png_path, dpi=160)

