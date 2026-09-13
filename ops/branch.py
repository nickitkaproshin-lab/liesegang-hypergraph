"""
ops/branch.py

Phi_branch: topological defect of the front.
C-shaped fracture in 2D Cartesian geometry.

Run:  python -m ops.branch
Output: figures/phi_branch.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_2d_cart, stable_dt


def run_branch(Lx=40.0, Ly=40.0, Nx=200, Ny=200,
               x0=20.0, y0=20.0, r_source=2.5,
               defect_sector=None, b_defect=0.3,
               Tmax=250.0, snapshot_times=(60.0, 120.0, 250.0),
               **kwargs):
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

    if defect_sector is not None:
        theta = np.mod(np.arctan2(Y - y0, X - x0), 2.0 * np.pi)
        t1 = np.mod(defect_sector[0], 2.0 * np.pi)
        t2 = np.mod(defect_sector[1], 2.0 * np.pi)
        mask = ((theta >= t1) & (theta <= t2)) if t1 < t2 \
               else ((theta >= t1) | (theta <= t2))
        b[mask] = b_defect

    src = ((X - x0)**2 + (Y - y0)**2) < r_source**2

    snapshots = {}
    snap_list = sorted(snapshot_times)
    next_snap = 0
    t = 0.0

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
        t += dt
        if next_snap < len(snap_list) and t >= snap_list[next_snap]:
            snapshots[snap_list[next_snap]] = c.copy()
            next_snap += 1

    return x, y, c, snapshots


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_branch: C-shaped fracture")

    x, y, c_base, snap_base = run_branch(defect_sector=None)
    x, y, c_branch, snap_branch = run_branch(
        defect_sector=(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi / 3.0),
        b_defect=0.3)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    vmax = max(c_base.max(), c_branch.max(), 0.5)

    ax = axes[0, 0]
    im = ax.imshow(c_base.T, extent=[0, 40, 0, 40], origin='lower',
                   cmap='hot', vmin=0, vmax=vmax)
    ax.set_title('BASE: closed rings'); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

    for i, ts in enumerate([60.0, 120.0, 250.0]):
        ax = axes[0, i + 1]
        if ts in snap_base:
            im = ax.imshow(snap_base[ts].T, extent=[0, 40, 0, 40],
                           origin='lower', cmap='hot', vmin=0, vmax=vmax)
            ax.set_title(f'BASE, t={ts:.0f}')
            plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(c_branch.T, extent=[0, 40, 0, 40], origin='lower',
                   cmap='hot', vmin=0, vmax=vmax)
    ax.set_title('Phi_branch: C-fracture'); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

    for i, ts in enumerate([60.0, 120.0, 250.0]):
        ax = axes[1, i + 1]
        if ts in snap_branch:
            im = ax.imshow(snap_branch[ts].T, extent=[0, 40, 0, 40],
                           origin='lower', cmap='hot', vmin=0, vmax=vmax)
            ax.set_title(f'Phi_branch, t={ts:.0f}')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('figures/phi_branch.png', dpi=150)
    plt.close()
    print("Saved figures/phi_branch.png")


if __name__ == "__main__":
    main()
