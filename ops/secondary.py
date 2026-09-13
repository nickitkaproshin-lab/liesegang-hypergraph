"""
ops/secondary.py

Phi_secondary: secondary banding inside primary bands.

Run:  python -m ops.secondary
Output: figures/phi_secondary.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, lap_1d, stable_dt


def run_secondary(alpha_sec=0.0, L=40.0, Nx=400, Tmax=400.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = L / (Nx - 1)
    dt = stable_dt(p, h_min=dx)
    Nt = int(Tmax / dt)

    x = np.linspace(0.0, L, Nx)
    a = np.zeros(Nx)
    b = np.full(Nx, p['b0'])
    c1 = np.zeros(Nx)
    c2 = np.zeros(Nx)

    for n in range(Nt):
        la = lap_1d(a, dx)
        lb = lap_1d(b, dx)

        sup = np.clip(a * b, 0.0, 10.0)
        sat1 = np.clip(1.0 - c1 / p['c_max'], 0.0, 1.0)
        R1 = (p['k_nuc'] * np.maximum(0.0, sup - p['K_nuc']) * sat1
              + p['k_auto'] * c1 * np.maximum(0.0, sup - p['K_sp']) * sat1)
        R2 = alpha_sec * c1 * a * (1.0 - c2 / p['c_max'])

        a += dt * (p['Da'] * la - R1 - R2)
        b += dt * (p['Db'] * lb - R1)
        c1 += dt * R1
        c2 += dt * R2
        a[0] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c1, 0.0, p['c_max'], out=c1)
        np.clip(c2, 0.0, p['c_max'], out=c2)

    return x, c1, c2


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_secondary: secondary banding")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for ax, alpha, title in zip(axes, [0.0, 0.5, 2.0],
                                ['BASE (alpha=0)', 'alpha=0.5', 'alpha=2.0']):
        x, c1, c2 = run_secondary(alpha_sec=alpha)
        ax.plot(x, c1, 'r-', lw=1.3, label='c1 (primary)')
        ax.plot(x, c2, 'b-', lw=1.3, label='c2 (secondary)')
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('c')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/phi_secondary.png', dpi=150)
    plt.close()
    print("Saved figures/phi_secondary.png")


if __name__ == "__main__":
    main()
