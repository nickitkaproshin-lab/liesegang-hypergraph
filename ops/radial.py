"""
ops/radial.py

Phi_radial: gradient of B from the wall.
Compression of band spacing near the wall.

Run:  python -m ops.radial
Output: figures/phi_radial.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_2d_cart, stable_dt


def run_radial(beta=0.0, Lx=40.0, Ly=20.0, Nx=200, Ny=100,
               Tmax=300.0, **kwargs):
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
    b = p['b0'] * (1.0 + beta * (Y / Ly)**2)
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
    print("Phi_radial: B gradient from the wall")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cases = [
        ("BASE: uniform b (beta=0)", dict(beta=0.0)),
        ("Phi_radial: beta=0.3", dict(beta=0.3)),
        ("Phi_radial: beta=0.6", dict(beta=0.6)),
    ]
    for ax, (title, kw) in zip(axes, cases):
        x, y, c = run_radial(**kw)
        im = ax.imshow(c.T, extent=[0, 40, 0, 20], origin='lower',
                       cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('figures/phi_radial.png', dpi=150)
    plt.close()
    print("Saved figures/phi_radial.png")


if __name__ == "__main__":
    main()
