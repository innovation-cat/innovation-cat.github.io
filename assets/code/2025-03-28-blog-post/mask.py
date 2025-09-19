# Re-run after state reset to generate the visual masks.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

torch.manual_seed(0)

def mask_dense(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool)

def mask_local(n: int, k: int) -> torch.Tensor:
    i = torch.arange(n).unsqueeze(1)
    j = torch.arange(n).unsqueeze(0)
    return (j >= i - k) & (j <= i + k)

def mask_stride(n: int, stride: int) -> torch.Tensor:
    i = torch.arange(n).unsqueeze(1)
    j = torch.arange(n).unsqueeze(0)
    return ((i - j).abs() % stride == 0)

def mask_block(n: int, block: int) -> torch.Tensor:
    i = torch.arange(n).unsqueeze(1)
    j = torch.arange(n).unsqueeze(0)
    return (i // block) == (j // block)

def mask_longformer(n: int, k: int, globals_idx: List[int]) -> torch.Tensor:
    m = mask_local(n, k)
    for g in globals_idx:
        m[:, g] = True
        m[g, :] = True
    return m

def mask_bigbird(n: int, k: int, globals_idx: List[int], r: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    m = mask_longformer(n, k, globals_idx)
    for i in range(n):
        rand_keys = torch.randperm(n)[:r]
        m[i, rand_keys] = True
    return m

def lsh_buckets(Q: torch.Tensor, n_hashes: int = 2, bucket_size: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    n, d = Q.shape
    planes = torch.randn(n_hashes, d)
    code = (Q @ planes.T > 0).to(torch.int64)
    weights = (2 ** torch.arange(n_hashes)).to(code.dtype)
    keys = (code * weights).sum(dim=1)
    uniq = keys.unique(sorted=True)
    buckets = []
    for u in uniq:
        idx = (keys == u).nonzero(as_tuple=False).flatten()
        for s in range(0, idx.numel(), bucket_size):
            buckets.append(idx[s:s + bucket_size])
    return buckets

def mask_reformer(n: int, d: int, n_hashes: int = 2, bucket_size: int = 12, seed: int = 0) -> torch.Tensor:
    Q = torch.randn(n, d)
    m = torch.zeros(n, n, dtype=torch.bool)
    for b in lsh_buckets(Q, n_hashes=n_hashes, bucket_size=bucket_size, seed=seed):
        m[b.unsqueeze(1), b.unsqueeze(0)] = True
    return m

def kmeans_labels(x: torch.Tensor, num_clusters: int = 6, iters: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    n, d = x.shape
    centers = x[torch.randperm(n)[:num_clusters]].clone()
    for _ in range(iters):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)
        for c in range(num_clusters):
            sel = (labels == c)
            if sel.any():
                centers[c] = x[sel].mean(dim=0)
    return labels

def mask_routing(n: int, d: int, num_clusters: int = 6, seed: int = 0) -> torch.Tensor:
    X = torch.randn(n, d)
    labels = kmeans_labels(X, num_clusters=num_clusters, seed=seed)
    m = torch.zeros(n, n, dtype=torch.bool)
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=False).flatten()
        m[idx.unsqueeze(1), idx.unsqueeze(0)] = True
    return m

def draw_mask(ax, mask: torch.Tensor, title: str):
    cmap = ListedColormap(["#cfcfcf", "#cfe8ff"])  # gray=0, light blue=1
    ax.imshow(mask.to(torch.int32), cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    n = mask.shape[0]
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, n-0.5)
    for g in range(n+1):
        ax.axhline(g-0.5, color="black", linewidth=0.4, alpha=0.6)
        ax.axvline(g-0.5, color="black", linewidth=0.4, alpha=0.6)

# Build and plot
n, d = 36, 64
k = 3
stride = 4
block = 6
globals_idx = [0, n-1]
r = 2
bucket = 12
clusters = 6

fig, axs = plt.subplots(2, 4, figsize=(12, 6))

masks = [
    ("Local (k=3)",            mask_local(n, k)),
    ("Stride (s=4)",           mask_stride(n, stride)),
    ("Block (b=6)",            mask_block(n, block)),
    ("Longformer (k=3,g={1, n})",   mask_longformer(n, k, globals_idx)),
    ("BigBird (k=3,g={1, n},r=2)",  mask_bigbird(n, k, globals_idx, r, seed=0)),
    ("Reformer (bucket=12)",   mask_reformer(n, d, n_hashes=2, bucket_size=bucket, seed=0)),
    ("Routing (clusters=6)",   mask_routing(n, d, num_clusters=clusters, seed=0)),
    ("Dense (baseline)",       mask_dense(n)),
]

for ax, (title, m) in zip(axs.flatten(), masks):
    draw_mask(ax, m, title)

plt.tight_layout()
out_path = "sparse_attention_masks_grid.png"
plt.savefig(out_path, dpi=180)
plt.show()

