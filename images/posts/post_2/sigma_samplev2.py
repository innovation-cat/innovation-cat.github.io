import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 设置参数 ---
# 与EDM论文保持一致的典型参数
sigma_min = 0.002
sigma_max = 80.0
# EDM Log-Normal分布的参数 (可以调整来观察变化)
P_mean = -1.2
P_std = 1.2
# 生成的样本数量
num_samples = 20000

# --- 2. 生成三种策略的样本 ---

# 策略1: 线性 Sigma (在 sigma 空间均匀采样)
samples_linear_sigma = np.random.uniform(sigma_min, sigma_max, num_samples)

# 策略2: 线性 log Sigma (Log-Uniform 分布)
log_sigma_min = np.log(sigma_min)
log_sigma_max = np.log(sigma_max)
samples_log_uniform = np.exp(np.random.uniform(log_sigma_min, log_sigma_max, num_samples))

# 策略3: log Sigma 高斯 (Log-Normal 分布, EDM策略)
log_samples_normal = np.random.normal(loc=P_mean, scale=P_std, size=num_samples)
samples_log_normal = np.exp(log_samples_normal)


# --- 3. 绘图 ---

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

fig, ax = plt.subplots(figsize=(14, 8))

# 使用 seaborn 的 histplot 绘制密度直方图
# alpha 值让直方图半透明，方便观察重叠部分
sns.histplot(samples_linear_sigma, bins=100, stat='density', alpha=0.6, label='策略1: 线性 Sigma (Uniform)', ax=ax, color='skyblue')
sns.histplot(samples_log_uniform, bins=100, stat='density', alpha=0.6, label='策略2: 线性 log Sigma (Log-Uniform)', ax=ax, color='salmon')
#sns.histplot(samples_log_normal, bins=100, stat='density', alpha=0.8, label='策略3: log Sigma 高斯 (EDM)', ax=ax, color='lightgreen', edgecolor='black', linewidth=0.5)

# 关键步骤：将X轴设置为对数尺度！
# 这样才能清晰地看到在小sigma值区域的分布情况
ax.set_xscale('log')

# 添加标题和标签
ax.set_title('三种不同 σ 采样策略的分布对比', fontsize=18, pad=20)
ax.set_xlabel('σ (噪声水平) [对数坐标轴]', fontsize=14)
ax.set_ylabel('采样密度', fontsize=14)

# 设置X轴范围，使其包含所有重要区域
ax.set_xlim(sigma_min * 0.5, sigma_max * 2)

# 显示图例
ax.legend(fontsize=12)

plt.show()