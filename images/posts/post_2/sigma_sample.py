# Sigma sampling visualization code (histogram + empirical CDF with thresholds)
# - Uses matplotlib only (no seaborn), single-plot per figure, default colors.
# - Saves figures and raw samples to /mnt/data for download/embedding in your blog.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Configurable parameters
# -----------------------
N = 500000                    # samples per strategy
sigma_min, sigma_max = 0.002, 80.0
mu, std = -1.2, 1.1             # log-normal parameters: log sigma ~ N(mu, std^2)
num_bins = 80                  # number of log-spaced bins for empirical density
thresholds = [1e-2, 1e-1, 1.0, 1e1]  # vertical markers on CDF plot

# -----------------------
# Reproducibility
# -----------------------
rng = np.random.default_rng(7)

# -----------------------
# Sampling helpers
# -----------------------
def sample_uniform_sigma(n, smin, smax, rng):
    return rng.uniform(low=smin, high=smax, size=n)

def sample_log_uniform_sigma(n, smin, smax, rng):
    log_min, log_max = np.log(smin), np.log(smax)
    return np.exp(rng.uniform(low=log_min, high=log_max, size=n))

def sample_truncated_lognormal(n, mu, sigma, smin, smax, rng):
    out = []
    batch = n
    while len(out) < n:
        s = rng.lognormal(mean=mu, sigma=sigma, size=batch)
        s = s[(s >= smin) & (s <= smax)]
        out.extend(s.tolist())
        batch = max(1000, n - len(out))
    return np.array(out[:n])

# -----------------------
# Draw samples
# -----------------------
sigma_linear = sample_uniform_sigma(N, sigma_min, sigma_max, rng)
sigma_log_uniform = sample_log_uniform_sigma(N, sigma_min, sigma_max, rng)
sigma_lognormal = sample_truncated_lognormal(N, mu, std, sigma_min, sigma_max, rng)



# -----------------------
# Empirical density (histogram with log-spaced bins)
# -----------------------
bins = np.logspace(np.log10(sigma_min), np.log10(sigma_max), num_bins)

plt.figure(figsize=(8,5))
plt.hist(sigma_linear, bins=bins, density=True, histtype='step', linewidth=2, label='Uniform in σ (linear)')
plt.hist(sigma_log_uniform, bins=bins, density=True, histtype='step', linewidth=2, label='Uniform in log σ')
plt.hist(sigma_lognormal, bins=bins, density=True, histtype='step', linewidth=2, label='Log-normal (truncated)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('σ (log scale)')
plt.ylabel('Empirical density (normalized)')
plt.title('Empirical sampling density from different p(σ) strategies')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
hist_path = "sigma_sampling_hist.png"
plt.savefig(hist_path, dpi=200)
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
rng = np.random.default_rng(7)

# Sigma range
sigma_min, sigma_max = 0.002, 80.0
N = 200_000

# 1) Uniform in sigma
sigma_linear = rng.uniform(low=sigma_min, high=sigma_max, size=N)

# 2) Uniform in log sigma
log_min, log_max = np.log(sigma_min), np.log(sigma_max)
sigma_log_uniform = np.exp(rng.uniform(low=log_min, high=log_max, size=N))

# 3) Log-normal (truncated)
mu, std = -1.2, 1.1
def sample_truncated_lognormal(n):
    out = []
    batch = n
    while len(out) < n:
        s = rng.lognormal(mean=mu, sigma=std, size=batch)
        s = s[(s >= sigma_min) & (s <= sigma_max)]
        out.extend(s.tolist())
        batch = max(1000, n - len(out))
    return np.array(out[:n])

sigma_lognormal = sample_truncated_lognormal(N)

# Common bins for linear and log scale
bins_linear = np.linspace(sigma_min, sigma_max, 200)
bins_log = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 80)

# Create two subplots: linear x-axis and log x-axis
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# Left: linear x-axis
axes[0].hist(sigma_linear, bins=bins_linear, density=True, histtype='step', label='Uniform in σ (linear)')
axes[0].hist(sigma_log_uniform, bins=bins_linear, density=True, histtype='step', label='Uniform in log σ')
axes[0].hist(sigma_lognormal, bins=bins_linear, density=True, histtype='step', label='Log-normal (truncated)')
axes[0].set_xlabel('σ (linear scale)')
axes[0].set_ylabel('Empirical density')
axes[0].set_title('Sampling distributions (linear axis)')
axes[0].legend()
axes[0].grid(True, ls='--', alpha=0.5)

# Right: log x-axis
axes[1].hist(sigma_linear, bins=bins_log, density=True, histtype='step', label='Uniform in σ (linear)')
axes[1].hist(sigma_log_uniform, bins=bins_log, density=True, histtype='step', label='Uniform in log σ')
axes[1].hist(sigma_lognormal, bins=bins_log, density=True, histtype='step', label='Log-normal (truncated)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('σ (log scale)')
axes[1].set_ylabel('Empirical density (normalized)')
axes[1].set_title('Sampling distributions (log-log axis)')
axes[1].legend()
axes[1].grid(True, which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.show()

'''