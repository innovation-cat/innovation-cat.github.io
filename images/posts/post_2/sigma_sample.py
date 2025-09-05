# Sigma sampling visualization code (histogram + empirical CDF with thresholds)
# - Uses matplotlib only (no seaborn), single-plot per figure, default colors.
# - Saves figures and raw samples to /mnt/data for download/embedding in your blog.
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Configurable parameters
# -----------------------
N = 500000                    # samples per strategy
sigma_min, sigma_max = 1e-3, 100.0
mu, std = 0.0, 1.1             # log-normal parameters: log sigma ~ N(mu, std^2)
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
plt.hist(sigma_linear, bins=bins, density=True, histtype='step', label='Uniform in σ (linear)')
plt.hist(sigma_log_uniform, bins=bins, density=True, histtype='step', label='Uniform in log σ')
plt.hist(sigma_lognormal, bins=bins, density=True, histtype='step', label='Log-normal (truncated)')
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
# -----------------------
# Empirical CDFs
# -----------------------
def empirical_cdf(samples, xs):
    s = np.sort(samples)
    # For each x in xs, compute fraction <= x
    return np.searchsorted(s, xs, side="right") / s.size

xs = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 2000)
cdf_lin = empirical_cdf(sigma_linear, xs)
cdf_logu = empirical_cdf(sigma_log_uniform, xs)
cdf_logn = empirical_cdf(sigma_lognormal, xs)

plt.figure(figsize=(8,5))
plt.plot(xs, cdf_lin, label='Uniform in σ (linear)')
plt.plot(xs, cdf_logu, label='Uniform in log σ')
plt.plot(xs, cdf_logn, label='Log-normal (truncated)')
for t in thresholds:
    plt.axvline(t, linestyle='--', linewidth=1)
plt.xscale('log')
plt.xlabel('σ (log scale)')
plt.ylabel('Empirical CDF')
plt.title('Empirical CDF of σ-sampling strategies')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
cdf_path = "/mnt/data/sigma_sampling_empirical_cdf.png"
plt.savefig(cdf_path, dpi=200)
plt.show()

# -----------------------
# Compute and print cumulative probabilities at thresholds
# -----------------------
def summarize_thresholds(samples, thresh_list):
    out = []
    for t in thresh_list:
        frac = (samples <= t).mean()
        out.append((t, frac))
    return out

summary_lin = summarize_thresholds(sigma_linear, thresholds)
summary_logu = summarize_thresholds(sigma_log_uniform, thresholds)
summary_logn = summarize_thresholds(sigma_lognormal, thresholds)

summary_lin, summary_logu, summary_logn, hist_path, cdf_path
'''