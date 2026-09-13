"""
ops/ads.py

Phi_ads: cumulative adsorption of A on the precipitate.
Da_eff(x) = Da / (1 + alpha * S(x)), where S(x) = int_0^x c(x') dx'.

Run:  python -m ops.ads
Output: figures/phi_ads.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, lap_1d, stable_dt, count_bands_1d


def run_ads(alpha=0.0, L=40.0, Nx=400, Tmax=400.0, **kwargs):
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
        S = np.cumsum(c) * dx
        Da_eff = p['Da'] / (1.0 + alpha * S)

        la = lap_1d(a, dx)
        lb = lap_1d(b, dx)
        R = reaction_rate(a, b, c,
                          p['K_nuc'], p['K_sp'],
                          p['k_nuc'], p['k_auto'], p['c_max'])

        a += dt * (Da_eff * la - R)
        b += dt * (p['Db'] * lb - R)
        c += dt * R
        a[0] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return x, c


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_ads: cumulative adsorption")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    cases = [
        ("BASE (alpha=0)", dict(alpha=0.0)),
        ("Phi_ads: alpha=0.05", dict(alpha=0.05)),
        ("Phi_ads: alpha=0.5", dict(alpha=0.5)),
    ]
    for ax, (title, kw) in zip(axes, cases):
        x, c = run_ads(**kw)
        bands = count_bands_1d(c)
        ax.plot(x, c, 'r-', lw=1.3)
        ax.set_title(f'{title}, N={len(bands)}')
        ax.set_xlabel('x'); ax.set_ylabel('c')
        ax.grid(True, alpha=0.3)
        if len(bands) >= 3:
            xb = x[np.array(bands)]
            print(f"  {title}: gaps = {np.round(np.diff(xb), 2)}")
    plt.tight_layout()
    plt.savefig('figures/phi_ads.png', dpi=150)
    plt.close()
    print("Saved figures/phi_ads.png")


if __name__ == "__main__":
    main()
