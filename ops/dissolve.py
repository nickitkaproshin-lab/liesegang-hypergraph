"""
ops/dissolve.py

Phi_dissolve: precipitation waves via soluble form S.
C -> S -> A.

Run:  python -m ops.dissolve
Output: figures/phi_dissolve.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, lap_1d, stable_dt


def run_dissolve(k_dis=0.0, L=40.0, Nx=400, Tmax=400.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dx = L / (Nx - 1)
    dt = stable_dt(p, h_min=dx)
    Nt = int(Tmax / dt)

    x = np.linspace(0.0, L, Nx)
    a = np.zeros(Nx)
    b = np.full(Nx, p['b0'])
    c = np.zeros(Nx)
    s = np.zeros(Nx)

    for n in range(Nt):
        la = lap_1d(a, dx)
        lb = lap_1d(b, dx)
        ls = lap_1d(s, dx)

        sup = np.clip(a * b, 0.0, 10.0)
        sat = np.clip(1.0 - c / p['c_max'], 0.0, 1.0)
        R = (p['k_nuc'] * np.maximum(0.0, sup - p['K_nuc']) * sat
             + p['k_auto'] * c * np.maximum(0.0, sup - p['K_sp']) * sat)
        R_dis = k_dis * c
        R_ret = k_dis * s

        a += dt * (p['Da'] * la - R + R_ret)
        b += dt * (p['Db'] * lb - R)
        c += dt * (R - R_dis)
        s += dt * (1.0 * ls + R_dis - R_ret)
        a[0] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)
        np.clip(s, 0.0, None, out=s)

    return x, c, s


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_dissolve: precipitation waves")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for ax, kd, title in zip(axes, [0.0, 0.05, 0.2],
                             ['BASE (k_dis=0)', 'k_dis=0.05', 'k_dis=0.2']):
        x, c, s = run_dissolve(k_dis=kd)
        ax.plot(x, c, 'r-', lw=1.3, label='c (precipitate)')
        ax.plot(x, s, 'b-', lw=1.3, label='s (soluble)')
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('c, s')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/phi_dissolve.png', dpi=150)
    plt.close()
    print("Saved figures/phi_dissolve.png")


if __name__ == "__main__":
    main()
