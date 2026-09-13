"""
ops/curv.py

Phi_curv: Liesegang eyes via phase boundary.

Run:  python -m ops.curv
Output: figures/phi_curv.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, stable_dt


def run_curv(R_drop=0.0, b_inside=1.0, R_max=20.0, Nr=400,
             Tmax=300.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dr = R_max / (Nr - 1)
    dt = stable_dt(p, h_min=dr)
    Nt = int(Tmax / dt)

    r = np.linspace(0.0, R_max, Nr)
    a = np.zeros(Nr)
    a[r < 2.0] = p['a_left']
    b = np.full(Nr, p['b0'])
    if R_drop > 0.0:
        b[r < R_drop] = b_inside
    c = np.zeros(Nr)

    inv_r = np.zeros(Nr); inv_r[1:] = 1.0 / r[1:]
    src = r < 2.0

    for n in range(Nt):
        a_pad = np.pad(a, 1, mode='edge')
        b_pad = np.pad(b, 1, mode='edge')
        lap_a = ((a_pad[2:] - 2.0 * a_pad[1:-1] + a_pad[:-2]) / dr**2
                 + inv_r * (a_pad[2:] - a_pad[:-2]) / (2.0 * dr))
        lap_b = ((b_pad[2:] - 2.0 * b_pad[1:-1] + b_pad[:-2]) / dr**2
                 + inv_r * (b_pad[2:] - b_pad[:-2]) / (2.0 * dr))

        R = reaction_rate(a, b, c,
                          p['K_nuc'], p['K_sp'],
                          p['k_nuc'], p['k_auto'], p['c_max'])

        a += dt * (p['Da'] * lap_a - R)
        b += dt * (p['Db'] * lap_b - R)
        c += dt * R
        a[src] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return r, c


def find_eye(r, c, c_thresh=0.1):
    mask = c > c_thresh
    if not mask.any():
        return None
    return r[np.argmax(mask)]


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_curv: Liesegang eyes")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cases = [
        ("BASE: uniform b", dict(R_drop=0.0, b_inside=1.0)),
        ("Phi_curv: R_drop=2.0", dict(R_drop=2.0, b_inside=0.0)),
        ("Phi_curv: R_drop=4.0", dict(R_drop=4.0, b_inside=0.0)),
    ]
    for ax, (title, kw) in zip(axes, cases):
        r, c = run_curv(**kw)
        eye = find_eye(r, c)
        ax.plot(r, c, 'r-', lw=1.5)
        if eye is not None:
            ax.axvline(eye, color='blue', linestyle='--', alpha=0.6,
                       label=f'R_eye={eye:.2f}')
            ax.legend()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('r'); ax.set_ylabel('c')
        ax.grid(True, alpha=0.3)
        print(f"  {title}: R_eye = {eye if eye is not None else 'n/a'}")
    plt.tight_layout()
    plt.savefig('figures/phi_curv.png', dpi=150)
    plt.close()
    print("Saved figures/phi_curv.png")


if __name__ == "__main__":
    main()
