"""
ops/compete.py

Phi_compete: two-precipitate dynamics.
Two cations A1, A2 compete for the common anion B.

Run:  python -m ops.compete
Output: figures/phi_compete.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, lap_1d, stable_dt


def run_compete(k2=0.0, L=40.0, Nx=400, Tmax=400.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = L / (Nx - 1)
    dt = stable_dt(p, h_min=dx)
    Nt = int(Tmax / dt)

    x = np.linspace(0.0, L, Nx)
    a1 = np.zeros(Nx)
    a2 = np.zeros(Nx)
    b = np.full(Nx, p['b0'])
    c1 = np.zeros(Nx)
    c2 = np.zeros(Nx)

    for n in range(Nt):
        la1 = lap_1d(a1, dx)
        la2 = lap_1d(a2, dx)
        lb = lap_1d(b, dx)

        sup1 = np.clip(a1 * b, 0.0, 10.0)
        sup2 = np.clip(a2 * b, 0.0, 10.0)
        sat1 = np.clip(1.0 - c1 / p['c_max'], 0.0, 1.0)
        sat2 = np.clip(1.0 - c2 / p['c_max'], 0.0, 1.0)
        R1 = (p['k_nuc'] * np.maximum(0.0, sup1 - p['K_nuc']) * sat1
              + p['k_auto'] * c1 * np.maximum(0.0, sup1 - p['K_sp']) * sat1)
        R2 = (p['k_nuc'] * np.maximum(0.0, sup2 - p['K_nuc']) * sat2
              + p['k_auto'] * c2 * np.maximum(0.0, sup2 - p['K_sp']) * sat2)

        a1 += dt * (p['Da'] * la1 - R1)
        a2 += dt * (p['Da'] * la2 - R2)
        b += dt * (p['Db'] * lb - R1 - R2)
        c1 += dt * R1
        c2 += dt * R2
        a1[0] = p['a_left']
        a2[0] = p['a_left'] * k2
        np.clip(a1, 0.0, None, out=a1)
        np.clip(a2, 0.0, None, out=a2)
        np.clip(b, 0.0, None, out=b)
        np.clip(c1, 0.0, p['c_max'], out=c1)
        np.clip(c2, 0.0, p['c_max'], out=c2)

    return x, c1, c2


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_compete: two-precipitate dynamics")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for ax, kk, title in zip(axes, [0.0, 0.5, 1.0],
                             ['BASE (k2=0)', 'k2=0.5', 'k2=1.0']):
        x, c1, c2 = run_compete(k2=kk)
        ax.plot(x, c1, 'r-', lw=1.3, label='C1')
        ax.plot(x, c2, 'b-', lw=1.3, label='C2')
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('c')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/phi_compete.png', dpi=150)
    plt.close()
    print("Saved figures/phi_compete.png")


if __name__ == "__main__":
    main()
