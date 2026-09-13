"""
ops/surf.py

Phi_surf: dendrites via random correlated nucleation field.
k_nuc(x,y) = k_nuc * (1 + eps * eta(x,y))
where eta is a Gaussian field with correlation length l.

Run:  python -m ops.surf
Output: figures/phi_surf.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label

from core import DEFAULTS, lap_2d_cart, stable_dt


def correlated_field(Nx, Ny, dx, dy, corr_length, seed=42):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((Nx, Ny))
    sx = corr_length / dx
    sy = corr_length / dy
    field = gaussian_filter(white, sigma=(sx, sy), mode='wrap')
    return (field - field.mean()) / (field.std() + 1e-12)


def run_surf(modulation=0.0, corr_length=2.0, seed=42,
             Lx=40.0, Ly=40.0, Nx=200, Ny=200,
             Tmax=250.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1)
    dt = stable_dt(p, h_min=min(dx, dy))
    Nt = int(Tmax / dt)

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    eta = correlated_field(Nx, Ny, dx, dy, corr_length, seed=seed)
    k_nuc_field = np.clip(p['k_nuc'] * (1.0 + modulation * eta), 0.0, None)

    a = np.zeros((Nx, Ny))
    b = np.full((Nx, Ny), p['b0'])
    c = np.zeros((Nx, Ny))
    src = X < 1.5

    for n in range(Nt):
        sup = np.clip(a * b, 0.0, 10.0)
        sat = np.clip(1.0 - c / p['c_max'], 0.0, 1.0)
        R = (k_nuc_field * np.maximum(0.0, sup - p['K_nuc']) * sat
             + p['k_auto'] * c * np.maximum(0.0, sup - p['K_sp']) * sat)
        a += dt * (p['Da'] * lap_2d_cart(a, dx, dy) - R)
        b += dt * (p['Db'] * lap_2d_cart(b, dx, dy) - R)
        c += dt * R
        a[src] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return x, y, c, eta


def count_fingers(c, c_min=0.3, min_size=10):
    mask = c > c_min
    _, n = label(mask)
    sizes = np.bincount(label(mask)[0].ravel())
    sizes[0] = 0
    return int((sizes > min_size).sum())


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_surf: dendrites via correlated nucleation field")

    cases = [
        ("BASE (eps=0)", dict(modulation=0.0)),
        ("eps=0.5, l=2.0", dict(modulation=0.5, corr_length=2.0)),
        ("eps=1.5, l=2.0", dict(modulation=1.5, corr_length=2.0)),
        ("eps=1.5, l=5.0", dict(modulation=1.5, corr_length=5.0)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for i, (title, kw) in enumerate(cases):
        x, y, c, eta = run_surf(**kw)
        n_fing = count_fingers(c)
        print(f"  {title}: fingers = {n_fing}")

        ax = axes[0, i]
        im = ax.imshow(c.T, extent=[0, 40, 0, 40], origin='lower',
                       cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{title} (N={n_fing})', fontsize=10)
        plt.colorbar(im, ax=ax)

        ax = axes[1, i]
        im = ax.imshow(eta.T, extent=[0, 40, 0, 40], origin='lower',
                       cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title('field eta', fontsize=10)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('figures/phi_surf.png', dpi=150)
    plt.close()
    print("Saved figures/phi_surf.png")


if __name__ == "__main__":
    main()
