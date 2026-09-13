"""
ops/ablation.py

Ablation protocol: remove one component of the kernel at a time.
A1: no autocatalysis (k_auto = 0)
A2: no depletion (Db = 0.5)
A3: mobile precipitate (Dc = 0.1)
A4: no source (a_left = 0)
A5: no nucleation (k_nuc = 0)

Run:  python -m ops.ablation
Output: figures/ablation_1d.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_1d, stable_dt, count_bands_1d


def run_ablation_1d(k_auto=None, Db=None, Dc=0.0, a_left=None,
                    k_nuc=None, L=40.0, Nx=400, Tmax=400.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)
    if k_auto is not None: p['k_auto'] = k_auto
    if Db     is not None: p['Db']     = Db
    if a_left is not None: p['a_left'] = a_left
    if k_nuc  is not None: p['k_nuc']  = k_nuc

    dx = L / (Nx - 1)
    dt = stable_dt(p, h_min=dx)
    Nt = int(Tmax / dt)

    x = np.linspace(0.0, L, Nx)
    a = np.zeros(Nx)
    b = np.full(Nx, p['b0'])
    c = np.zeros(Nx)

    for n in range(Nt):
        la = lap_1d(a, dx)
        lb = lap_1d(b, dx)
        lc = lap_1d(c, dx)
        R = reaction_rate(a, b, c,
                          p['K_nuc'], p['K_sp'],
                          p['k_nuc'], p['k_auto'], p['c_max'])
        a += dt * (p['Da'] * la - R)
        b += dt * (p['Db'] * lb - R)
        c += dt * (Dc * lc + R)
        a[0] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return x, c


def main():
    os.makedirs('figures', exist_ok=True)
    print("Ablation protocol (1D)")

    cases = [
        ("BASE (kernel)",         dict()),
        ("A1: no autocatalysis",  dict(k_auto=0.0)),
        ("A2: no depletion",      dict(Db=0.5)),
        ("A3: mobile precipitate",dict(Dc=0.1)),
        ("A4: no source",         dict(a_left=0.0)),
        ("A5: no nucleation",     dict(k_nuc=0.0)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, (title, kw) in zip(axes.flat, cases):
        x, c = run_ablation_1d(**kw)
        bands = count_bands_1d(c)
        N = len(bands)
        print(f"  {title}: N = {N}")
        ax.plot(x, c, 'r-', lw=1.3)
        ax.set_title(f'{title} (N={N})', fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('c')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/ablation_1d.png', dpi=150)
    plt.close()
    print("Saved figures/ablation_1d.png")


if __name__ == "__main__":
    main()
