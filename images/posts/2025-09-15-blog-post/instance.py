import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Parameters
n_points = 100

# Means farther apart
mean_left = torch.tensor([0.0, 0.0])
mean_right = torch.tensor([6.0, 0.0])

# Elliptical but vertical (taller in y-axis, narrower in x-axis)
cov_left = torch.diag(torch.tensor([0.02, 0.25]))
cov_right = torch.diag(torch.tensor([0.015, 0.22]))

# Sample
left_points = torch.distributions.MultivariateNormal(mean_left, cov_left).sample((n_points,))
right_points = torch.distributions.MultivariateNormal(mean_right, cov_right).sample((n_points,))

# Plot
plt.figure(figsize=(9, 4))
plt.scatter(left_points[:, 0], left_points[:, 1],  s=20, alpha=0.12)
plt.scatter(right_points[:, 0], right_points[:, 1],  s=20, alpha=0.12)

# Gray connections
for i in range(n_points):
    plt.plot([left_points[i, 0], right_points[i, 0]],
             [left_points[i, 1], right_points[i, 1]],
             color="gray", alpha=0.45, linewidth=0.9)

#plt.legend()
plt.axis("equal")
plt.title("Vertically Elongated Elliptical Gaussian Clouds (Gray Connections)")
plt.tight_layout()
plt.show()
