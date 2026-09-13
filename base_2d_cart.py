"""
base_2d_cart.py

BASE 2D Cartesian: flat front source.

Run:  python base_2d_cart.py
Output: figures/base_2d_cart.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_2d_cart, stable_dt


def run_base_2d_cart(Lx=40.0, Ly=40.0, Nx=150, Ny=150,
                     Tmax=200.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = stable_dt(p, h_min=min(dx, dy))
    Nt = int(Tmax / dt)

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    a = np.zeros((Nx, Ny))
    b = np.full((Nx, Ny), p['b0'])
    c = np.zeros((Nx, Ny))
    src = X < 1.5

    c_history = []
    save_every = max(1, Nt // 100)

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
        if n % save_every == 0:
            c_history.append(c.copy())

    return x, y, c, np.array(c_history)


def main():
    os.makedirs('figures', exist_ok=True)
    print("BASE 2D Cartesian")

    x, y, c, c_hist = run_base_2d_cart()
    print(f"  c_max = {c.max():.2f}, c_sum = {c.sum():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    snapshots = [c, c_hist[len(c_hist)//2], c_hist[-1]]
    titles = ['Final', 'Middle', 'Last']
    for ax, cc, tt in zip(axes, snapshots, titles):
        im = ax.imshow(cc.T, extent=[0, 40, 0, 40],
                       origin='lower', cmap='hot')
        ax.set_title(tt); ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('figures/base_2d_cart.png', dpi=150)
    plt.close()
    print("Saved figures/base_2d_cart.png")


if __name__ == "__main__":
    main()
