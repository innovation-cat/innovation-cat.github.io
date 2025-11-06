import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set font for better visualization
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Create grid
x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Define multimodal distribution function
def create_multimodal_distribution(means, covs, weights, grid_pos):
    """Create multimodal Gaussian mixture distribution"""
    density = np.zeros(grid_pos.shape[:2])
    for mean, cov, weight in zip(means, covs, weights):
        density += weight * multivariate_normal(mean, cov).pdf(grid_pos)
    return density / np.sum(weights)

# p(x): Unconditional distribution - 5 high-density regions
means_uncond = [
    [-3, -2], [3, 2], [-2, 3], [2, -3], [0, 0],  # 5 high-density peaks
    [-1, -2], [1, 1], [-1.5, 1.5], [2.5, -1], [-2, -2.5]  # 5 medium/low density regions
]

covs_uncond = [
    [[0.9, 0.2], [0.2, 0.7]], [[0.8, -0.15], [-0.15, 0.9]],
    [[1.0, 0.25], [0.25, 0.6]], [[0.7, -0.2], [-0.2, 0.8]],
    [[1.2, 0.1], [0.1, 1.0]], 
    [[1.5, 0.3], [0.3, 1.2]], [[1.3, -0.1], [-0.1, 1.1]],
    [[1.4, 0.2], [0.2, 0.9]], [[1.1, 0.0], [0.0, 1.3]],
    [[1.6, 0.25], [0.25, 1.1]]
]

weights_uncond = [2.0, 1.8, 1.7, 1.6, 2.2, 0.8, 0.9, 0.7, 0.6, 0.5]

p_x = create_multimodal_distribution(means_uncond, covs_uncond, weights_uncond, pos)

# p(x|c): Conditional distribution - 4 high-density regions that are subsets of p(x)'s high-density regions
means_cond = [
    [-3, -2], [3, 2], [-2, 3], [0, 0],  # 4 high-density peaks (subset of p(x)'s peaks)
    [-1, -2], [1, 1]  # 2 lower density regions
]

covs_cond = [
    [[0.6, 0.15], [0.15, 0.5]], [[0.5, -0.1], [-0.1, 0.7]],
    [[0.7, 0.2], [0.2, 0.4]], [[0.9, 0.08], [0.08, 0.8]],
    [[1.2, 0.25], [0.25, 0.9]], [[1.0, -0.08], [-0.08, 0.8]]
]

weights_cond = [2.2, 2.0, 1.9, 2.1, 0.6, 0.5]

p_x_given_c = create_multimodal_distribution(means_cond, covs_cond, weights_cond, pos)

# Create the single comparison figure
fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Enhanced contrast colormaps
cmap_bg = 'Blues'
cmap_fg = 'Reds'

# Plot p(x) as background
levels_bg = np.linspace(0.005, p_x.max(), 50)
contour_bg = ax.contourf(X, Y, p_x, levels=levels_bg, cmap=cmap_bg, alpha=0.6)
bg_contours = ax.contour(X, Y, p_x, levels=levels_bg[::6], colors='darkblue', linewidths=0.4, alpha=0.4)

# Plot p(x|c) as foreground
levels_fg = np.linspace(0.005, p_x_given_c.max(), 45)
contour_fg = ax.contourf(X, Y, p_x_given_c, levels=levels_fg, cmap=cmap_fg, alpha=0.8)
fg_contours = ax.contour(X, Y, p_x_given_c, levels=levels_fg[::5], colors='darkred', linewidths=0.7, alpha=0.6)

# Add labels and title
ax.set_xlabel('Feature Dimension 1', fontsize=16, fontweight='bold')
ax.set_ylabel('Feature Dimension 2', fontsize=16, fontweight='bold')
ax.set_title('Manifold Relationship: p(x) is a Superset of p(x|c)\n' + 
             'p(x) has 5 High-Density Regions, p(x|c) has 4 High-Density Regions', 
             fontsize=18, fontweight='bold', pad=25)

# Add grid
ax.grid(True, alpha=0.15, linestyle='--')

# Mark p(x) high-density regions (5 regions)
#p_x_high_density_points = [(-3, -2), (3, 2), (-2, 3), (2, -3), (0, 0)]
p_x_high_density_points = []
for i, (px, py) in enumerate(p_x_high_density_points):
    ax.plot(px, py, 'o', markersize=14, markeredgecolor='white', 
            markerfacecolor='blue', alpha=0.9, markeredgewidth=2)
    ax.annotate(f'p(x) Peak {i+1}', (px, py), xytext=(px+0.4, py+0.4), 
                fontsize=11, fontweight='bold', color='darkblue')

# Mark p(x|c) high-density regions (4 regions - subset of p(x))
#p_xc_high_density_points = [(-3, -2), (3, 2), (-2, 3), (0, 0)]
p_xc_high_density_points = []
for i, (px, py) in enumerate(p_xc_high_density_points):
    ax.plot(px, py, 's', markersize=16, markeredgecolor='yellow', 
            markerfacecolor='red', alpha=1.0, markeredgewidth=3)
    ax.annotate(f'p(x|c) Core {i+1}', (px, py), xytext=(px+0.5, py-0.8), 
                fontsize=12, fontweight='bold', color='darkred')

# Highlight the p(x) peak that is NOT in p(x|c)
#ax.plot(2, -3, 'o', markersize=14, markeredgecolor='white', 
#        markerfacecolor='blue', alpha=0.9, markeredgewidth=2)
#ax.annotate('p(x) Peak 4\n(not in p(x|c))', (2, -3), xytext=(2.5, -4), 
#            fontsize=11, fontweight='bold', color='darkblue',
#            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

# Add density relationship annotations
#ax.annotate('High-Density Regions:\n• Where p(x) is high,\n  p(x|c) is also high', 
#            xy=(-3, -2), xytext=(-4.5, -4),
#            arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.8),
#            fontsize=12, fontweight='bold', color='green',
#            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

#ax.annotate('Low-Density Regions:\n• Where p(x) is low,\n  p(x|c) is also low', 
#            xy=(4, 4), xytext=(2.5, 4.2),
#            arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0.8),
#            fontsize=12, fontweight='bold', color='purple',
#            bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.6, label='p(x) - Base Manifold (5 peaks)'),
    Patch(facecolor='red', alpha=0.8, label='p(x|c) - Conditional (4 peaks)'),
    Patch(facecolor='blue', alpha=0.9, label='p(x) High-Density Peaks'),
    Patch(facecolor='red', alpha=1.0, label='p(x|c) High-Density Cores'),
    Patch(facecolor='lightblue', alpha=0.7, label='p(x) Exclusive Regions')
]
#ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95)

# Add colorbars
plt.colorbar(contour_bg, ax=ax, label='p(x) Probability Density', shrink=0.8, pad=0.02)
plt.colorbar(contour_fg, ax=ax, label='p(x|c) Probability Density', shrink=0.8, pad=0.15)

# Add mathematical relationship text
'''
ax.text(-4.8, -4.8, 
        'Mathematical Relationship:\n'
        '• p(x|c) ⊆ p(x) (p(x|c) is a subset of p(x))\n'
        '• High-density regions of p(x|c) are subsets of high-density regions of p(x)\n'
        '• Low-density regions align between both distributions',
        fontsize=12, style='italic',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
'''

plt.tight_layout()
plt.show()