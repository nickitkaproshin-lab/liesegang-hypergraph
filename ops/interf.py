"""
ops/interf.py

Phi_interf: anastomoses via two sources of A.

Run:  python -m ops.interf
Output: figures/phi_interf.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_1d, stable_dt, count_bands_1d


def run_interf(two_sources=False, L=40.0, Nx=400, Tmax=400.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

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
        R = reaction_rate(a, b, c,
                          p['K_nuc'], p['K_sp'],
                          p['k_nuc'], p['k_auto'], p['c_max'])
        a += dt * (p['Da'] * la - R)
        b += dt * (p['Db'] * lb - R)
        c += dt * R
        a[0] = p['a_left']
        if two_sources:
            a[-1] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return x, c


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_interf: anastomoses")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, ts, title in zip(axes, [False, True],
                             ['BASE (one source)', 'Phi_interf (two sources)']):
        x, c = run_interf(two_sources=ts)
        bands = count_bands_1d(c)
        ax.plot(x, c, 'r-', lw=1.3)
        ax.set_title(f'{title}, N={len(bands)}')
        ax.set_xlabel('x'); ax.set_ylabel('c')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/phi_interf.png', dpi=150)
    plt.close()
    print("Saved figures/phi_interf.png")


if __name__ == "__main__":
    main()
