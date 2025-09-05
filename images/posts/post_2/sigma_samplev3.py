import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Define sigma range
sigma_min, sigma_max = 0.002, 80
sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 10000)

# Linear uniform distribution (normalized)
p_linear = np.ones_like(sigmas)
p_linear /= np.trapz(p_linear, sigmas)

# Log-uniform distribution (1/sigma, normalized)
p_log_uniform = 1 / sigmas
p_log_uniform /= np.trapz(p_log_uniform, sigmas)

# Log-normal distribution (choose mean=log(1.0), std=1.0)
mu, std = 0, 1.0
p_log_normal = lognorm.pdf(sigmas, s=std, scale=np.exp(mu))
p_log_normal /= np.trapz(p_log_normal, sigmas)

# Plot
plt.figure(figsize=(8,5))
plt.plot(sigmas, p_linear, label="Uniform in σ (linear)")
plt.plot(sigmas, p_log_uniform, label="Uniform in log σ")
plt.plot(sigmas, p_log_normal, label="Log-normal")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("σ (noise scale, log axis)")
plt.ylabel("p(σ) (normalized density, log axis)")
plt.title("Comparison of σ-sampling strategies")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

plt.show()
