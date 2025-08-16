# Plot an "area view" diagram for AB4 (Adams–Bashforth 4):
# - Build a cubic Lagrange interpolant through past slopes (f_{n-3}, f_{n-2}, f_{n-1}, f_n)
# - Shade the integral over [t_n, t_{n+1}]
# - Single figure, matplotlib only, no explicit colors

import numpy as np
import matplotlib.pyplot as plt

def lagrange_poly(x, xp, yp):
    x = np.asarray(x)
    xp = np.asarray(xp)
    yp = np.asarray(yp)
    L = np.zeros_like(x, dtype=float)
    n = len(xp)
    for i in range(n):
        li = np.ones_like(x, dtype=float)
        for j in range(n):
            if j == i:
                continue
            li *= (x - xp[j]) / (xp[i] - xp[j])
        L += yp[i] * li
    return L

# Uniform time grid for illustration
tnm3, tnm2, tnm1, tn, tnp1 = 0.0, 1.0, 2.0, 3.0, 4.0

# Sample slope values at past nodes (arbitrary but smooth-ish for visualization)
f_samples4 = np.array([0.6, 0.9, 1.2, 1.7])  # f_{n-3}, f_{n-2}, f_{n-1}, f_n
t_samples4 = np.array([tnm3, tnm2, tnm1, tn])

# Interpolant over a wider plotting range
x_plot = np.linspace(tnm3-0.2, tnp1+0.2, 600)
f_interp4 = lagrange_poly(x_plot, t_samples4, f_samples4)

plt.figure(figsize=(8,4.5))
plt.plot(x_plot, f_interp4, linewidth=2.0, label='Cubic interpolant of past slopes')
plt.scatter(t_samples4, f_samples4, marker='o', label=r'$f_{n-3}, f_{n-2}, f_{n-1}, f_n$')

# Shade the area over [t_n, t_{n+1}]
mask = (x_plot >= tn) & (x_plot <= tnp1)
plt.fill_between(x_plot[mask], 0, f_interp4[mask], alpha=0.3, label=r'$\int_{t_n}^{t_{n+1}}\tilde f(s)\,ds$')

# Vertical markers
for t_val, lab in [(tnm3, r'$t_{n-3}$'), (tnm2, r'$t_{n-2}$'), (tnm1, r'$t_{n-1}$'), (tn, r'$t_n$'), (tnp1, r'$t_{n+1}$')]:
    plt.axvline(t_val, linewidth=1.0, linestyle='--')
    plt.text(t_val, min(f_interp4)+0.15, lab, ha='center', va='top')

plt.title('Adams–Bashforth 4 (AB4): integrate a cubic interpolant of past slopes')
plt.xlabel('s')
plt.ylabel('slope ~ f(s, x(s))')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig('./area_AB4.png', dpi=150)
plt.close()
