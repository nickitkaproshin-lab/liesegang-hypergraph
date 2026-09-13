"""
base_1d.py

BASE 1D: parallel bands.

Run:  python base_1d.py
Output: figures/base_1d.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_1d, stable_dt, count_bands_1d


def run_base_1d(L=40.0, Nx=400, Tmax=400.0, **kwargs):
    """Run 1D model, return x, c, c_history, Tmax."""
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = L / (Nx - 1)
    dt = stable_dt(p, h_min=dx)
    Nt = int(Tmax / dt)

    x = np.linspace(0, L, Nx)
    a = np.zeros(Nx)
    b = np.full(Nx, p['b0'])
    c = np.zeros(Nx)

    c_history = []
    save_every = max(1, Nt // 200)

    for n in range(Nt):
        R = reaction_rate(a, b, c,
                          p['K_nuc'], p['K_sp'],
                          p['k_nuc'], p['k_auto'], p['c_max'])
        a += dt * (p['Da'] * lap_1d(a, dx) - R)
        b += dt * (p['Db'] * lap_1d(b, dx) - R)
        c += dt * R
        a[0] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

        if n % save_every == 0:
            c_history.append(c.copy())

    return x, c, np.array(c_history), Tmax


def main():
    os.makedirs('figures', exist_ok=True)
    print("BASE 1D: parallel bands")

    x, c, c_hist, Tmax = run_base_1d()
    bands = count_bands_1d(c)
    N = len(bands)

    print(f"  N = {N} bands")
    if N >= 3:
        x_bands = x[np.array(bands)]
        gaps = np.diff(x_bands)
        print(f"  positions: {np.round(x_bands, 2)}")
        print(f"  gaps:      {np.round(gaps, 2)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(x, c, 'r-', lw=1.3)
    ax.set_xlabel('x'); ax.set_ylabel('c')
    ax.set_title(f'BASE 1D: c(x), N = {N}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    im = ax.imshow(c_hist.T, aspect='auto', origin='lower',
                   extent=[0, Tmax, 0, x[-1]], cmap='hot')
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_title('c(x, t)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('figures/base_1d.png', dpi=150)
    plt.close()
    print("Saved figures/base_1d.png")


if __name__ == "__main__":
    main()
