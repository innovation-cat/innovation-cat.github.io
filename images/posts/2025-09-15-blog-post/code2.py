import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(0)

# Lighter settings to avoid timeout
N = 500
T = 50

mean_left  = torch.tensor([0.0, 0.0])
mean_right = torch.tensor([6.0, 0.0])
cov_left  = torch.diag(torch.tensor([0.02, 0.25]))
cov_right = torch.diag(torch.tensor([0.015, 0.22]))

left  = torch.distributions.MultivariateNormal(mean_left,  cov_left ).sample((N,))
right = torch.distributions.MultivariateNormal(mean_right, cov_right).sample((N,))

u = right - left

t_grid = torch.linspace(0.0, 1.0, T)
x_t = (1 - t_grid.view(T, 1, 1)) * left + t_grid.view(T, 1, 1) * right

def v_hat(x_query: torch.Tensor, t_index: int, h: float = 0.2):
    xt = x_t[t_index]
    diff = xt - x_query
    d2 = (diff**2).sum(dim=1)
    w = torch.exp(-d2 / (2*h*h)) + 1e-12
    w = w / w.sum()
    return (w.unsqueeze(1) * u).sum(dim=0)

def integrate_curve(x0: torch.Tensor):
    traj = [x0.clone()]
    for k in range(T - 1):
        v = v_hat(traj[-1], k)
        dt = t_grid[k+1] - t_grid[k]
        traj.append(traj[-1] + v * dt)
    return torch.stack(traj, dim=0)

seed_ys = torch.linspace(-1.1, 1.1, 7)
seeds = [torch.tensor([mean_left[0], y]) for y in seed_ys]
curves = [integrate_curve(s) for s in seeds]

mid_idx = T // 2
xs = torch.linspace(-0.5, 6.5, 12)
ys = torch.linspace(-1.2, 1.2, 10)
grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)

# Vectorized quiver estimation by batching to speed up
def v_hat_batch(points: torch.Tensor, t_index: int, h: float = 0.2):
    xt = x_t[t_index]  # (N,2)
    P = points.shape[0]
    # Compute squared distances between all points and xt efficiently
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = (points**2).sum(dim=1, keepdim=True)         # (P,1)
    b2 = (xt**2).sum(dim=1).unsqueeze(0)              # (1,N)
    ab = points @ xt.T                                 # (P,N)
    d2 = a2 + b2 - 2*ab                                # (P,N)
    w = torch.exp(-d2 / (2*h*h)) + 1e-12               # (P,N)
    w = w / w.sum(dim=1, keepdim=True)                 # (P,N)
    V = w @ u                                          # (P,2)
    return V

V = v_hat_batch(grid, mid_idx)

# Plot
plt.figure(figsize=(9, 4.5))
plt.scatter(left[:, 0], left[:, 1], s=20, alpha=0.52)
plt.scatter(right[:, 0], right[:, 1], s=20, alpha=0.52)

# Gray connections
#for i in range(N):
#    plt.plot([left[i, 0], right[i, 0]],
#             [left[i, 1], right[i, 1]],
#             color="gray", alpha=0.45, linewidth=0.9)
             
#plt.title("Vertically Elongated Elliptical Gaussian Clouds (Gray Connections)")
#plt.tight_layout()
#plt.show()
#exit()
             
for tr in curves:
    plt.plot(tr[:, 0], tr[:, 1], linewidth=2.0, alpha=0.9)

Vn = V / (V.norm(dim=1, keepdim=True) + 1e-12)
plt.quiver(grid[:, 0], grid[:, 1], Vn[:, 0], Vn[:, 1], angles='xy', scale_units='xy', scale=12, alpha=0.6)

plt.title("Curved Mean Velocity Field in Flow Matching (Straight Sample Paths, Curved Mean Flow)")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()
