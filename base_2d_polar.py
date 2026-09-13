"""
base_2d_polar.py

BASE 2D radial: concentric rings.

Run:  python base_2d_polar.py
Output: figures/rings_2d.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, reaction_rate, stable_dt, count_rings


def run_base_2d(R_max=20.0, Nr=400, Tmax=400.0, r_source=3.0, **kwargs):
    """Axisymmetric 2D model. Returns r, c, c_history, Tmax."""
    p = dict(DEFAULTS)
    p.update(kwargs)

    dr = R_max / (Nr - 1)
    dt = stable_dt(p, h_min=dr)
    Nt = int(Tmax / dt)

    r = np.linspace(0.0, R_max, Nr)
    a = np.zeros(Nr)
    b = np.full(Nr, p['b0'])
    c = np.zeros(Nr)

    inv_r = np.zeros(Nr)
    inv_r[1:] = 1.0 / r[1:]

    src = r < r_source

    c_history = []
    save_every = max(1, Nt // 200)

    for n in range(Nt):
        # Axisymmetric polar Laplacian (no theta terms)
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

        if n % save_every == 0:
            c_history.append(c.copy())

    return r, c, np.array(c_history), Tmax


def main():
    os.makedirs('figures', exist_ok=True)
    print("BASE 2D radial: concentric rings")

    r, c, c_hist, Tmax = run_base_2d()
    rings = count_rings(c)
    N = len(rings)
    print(f"  N = {N} rings")
    if N >= 3:
        r_rings = r[np.array(rings)]
        print(f"  radii: {np.round(r_rings, 2)}")
        print(f"  gaps:  {np.round(np.diff(r_rings), 2)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(r, c, 'r-', lw=1.3)
    ax.set_xlabel('r'); ax.set_ylabel('c')
    ax.set_title(f'Radial profile, N = {N}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    R_max = r[-1]
    X, Y = np.meshgrid(np.linspace(-R_max, R_max, 300),
                       np.linspace(-R_max, R_max, 300))
    Rr = np.sqrt(X**2 + Y**2)
    C2 = np.interp(Rr, r, c)
    ax.imshow(C2, extent=[-R_max, R_max, -R_max, R_max],
              origin='lower', cmap='hot')
    ax.set_title('2D view')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[2]
    im = ax.imshow(c_hist.T, aspect='auto', origin='lower',
                   extent=[0, c_hist.shape[0], 0, R_max], cmap='hot')
    ax.set_xlabel('snapshot'); ax.set_ylabel('r')
    ax.set_title('c(r, t)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('figures/rings_2d.png', dpi=150)
    plt.close()
    print("Saved figures/rings_2d.png")


if __name__ == "__main__":
    main()
