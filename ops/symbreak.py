"""
ops/symbreak.py

Phi_symbreak: speckled patterns via 2D noise in b.

Run:  python -m ops.symbreak
Output: figures/phi_symbreak.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_2d_cart, stable_dt


def run_symbreak(noise=0.0, seed=42, Lx=40.0, Ly=40.0,
                 Nx=150, Ny=150, Tmax=200.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1)
    dt = stable_dt(p, h_min=min(dx, dy))
    Nt = int(Tmax / dt)

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    a = np.zeros((Nx, Ny))
    b = 1.0 + noise * rng.standard_normal((Nx, Ny))
    b = np.clip(b, 0.1, None)
    c = np.zeros((Nx, Ny))
    src = X < 1.5

    for n in range(Nt):
        R = reaction_rate(a, b, c,
                          p['K_nuc'], p['K_sp'],
                          p['k_nuc'], p['k_auto'], p['c_max'])
        a += dt * (p['Da'] * lap_2d_cart(a, dx, dy) - R)
        b += dt * (p['Db'] * lap_2d_cart(b, dx, dy) - R)
        c += dt * R
        a[src] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return x, y, c


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_symbreak: speckled patterns")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, noise, title in zip(axes, [0.0, 0.2, 0.5],
                                ['BASE (noise=0)', 'noise=0.2', 'noise=0.5']):
        x, y, c = run_symbreak(noise=noise)
        im = ax.imshow(c.T, extent=[0, 40, 0, 40], origin='lower', cmap='hot')
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('figures/phi_symbreak.png', dpi=150)
    plt.close()
    print("Saved figures/phi_symbreak.png")


if __name__ == "__main__":
    main()
