"""
ops/polygon.py

Phi_polygon: dislocations via angular anisotropy
k_nuc(theta) = k_nuc * (1 + eps * cos(n * theta))
in polar coordinates.

Run:  python -m ops.polygon
Output: figures/phi_polygon.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import DEFAULTS, stable_dt


def run_polygon(epsilon=0.0, n_fold=6,
                R_max=20.0, Nr=200, Ntheta=180,
                Tmax=300.0, **kwargs):
    p = dict(DEFAULTS)
    p.update(kwargs)

    dr = R_max / (Nr - 1)
    dtheta = 2.0 * np.pi / Ntheta
    dt = stable_dt(p, h_min=min(dr, R_max * dtheta))
    Nt = int(Tmax / dt)

    r = np.linspace(0.0, R_max, Nr)
    theta = np.linspace(0.0, 2.0 * np.pi, Ntheta, endpoint=False)
    R, T = np.meshgrid(r, theta, indexing='ij')

    k_nuc_field = p['k_nuc'] * (1.0 + epsilon * np.cos(n_fold * T))

    a = np.zeros((Nr, Ntheta))
    b = np.full((Nr, Ntheta), p['b0'])
    c = np.zeros((Nr, Ntheta))
    src = R < 2.0

    for n in range(Nt):
        # Polar Laplacian
        Nr_, Nt_ = a.shape
        inv_r = np.zeros(Nr_); inv_r[1:] = 1.0 / r[1:]
        inv_r2 = np.zeros(Nr_); inv_r2[1:] = 1.0 / r[1:]**2
        a_pad_r = np.pad(a, ((1, 1), (0, 0)), mode='edge')
        a_pad_t = np.pad(a, ((0, 0), (1, 1)), mode='wrap')
        b_pad_r = np.pad(b, ((1, 1), (0, 0)), mode='edge')
        b_pad_t = np.pad(b, ((0, 0), (1, 1)), mode='wrap')

        lap_a = ((a_pad_r[2:, :] - 2.0 * a + a_pad_r[:-2, :]) / dr**2
                 + inv_r[:, None] * (a_pad_r[2:, :] - a_pad_r[:-2, :]) / (2.0 * dr)
                 + inv_r2[:, None] * (a_pad_t[:, 2:] - 2.0 * a + a_pad_t[:, :-2]) / dtheta**2)
        lap_b = ((b_pad_r[2:, :] - 2.0 * b + b_pad_r[:-2, :]) / dr**2
                 + inv_r[:, None] * (b_pad_r[2:, :] - b_pad_r[:-2, :]) / (2.0 * dr)
                 + inv_r2[:, None] * (b_pad_t[:, 2:] - 2.0 * b + b_pad_t[:, :-2]) / dtheta**2)

        sup = np.clip(a * b, 0.0, 10.0)
        sat = np.clip(1.0 - c / p['c_max'], 0.0, 1.0)
        Rxn = (k_nuc_field * np.maximum(0.0, sup - p['K_nuc']) * sat
               + p['k_auto'] * c * np.maximum(0.0, sup - p['K_sp']) * sat)

        a += dt * (p['Da'] * lap_a - Rxn)
        b += dt * (p['Db'] * lap_b - Rxn)
        c += dt * Rxn
        a[src] = p['a_left']
        np.clip(a, 0.0, None, out=a)
        np.clip(b, 0.0, None, out=b)
        np.clip(c, 0.0, p['c_max'], out=c)

    return r, theta, c


def angular_fourier(c, r, theta, r_lo=5.0, r_hi=15.0):
    mask = (r >= r_lo) & (r <= r_hi)
    if mask.sum() < 2:
        return None
    c_theta = c[mask, :].mean(axis=0)
    fft = np.fft.rfft(c_theta - c_theta.mean())
    return np.abs(fft)**2


def main():
    os.makedirs('figures', exist_ok=True)
    print("Phi_polygon: dislocations via angular anisotropy")

    cases = [
        ("Isotropic (eps=0)", dict(epsilon=0.0)),
        ("Trigonal (eps=0.3, n=3)", dict(epsilon=0.3, n_fold=3)),
        ("Hexagonal (eps=0.3, n=6)", dict(epsilon=0.3, n_fold=6)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (title, kw) in enumerate(cases):
        r, theta, c = run_polygon(**kw)
        power = angular_fourier(c, r, theta)

        ax = axes[0, i]
        im = ax.imshow(c.T, extent=[0, 20, 0, 2*np.pi], origin='lower',
                       cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title); ax.set_xlabel('r'); ax.set_ylabel('theta')
        plt.colorbar(im, ax=ax)

        ax = axes[1, i]
        if power is not None:
            n_modes = np.arange(len(power))
            ax.bar(n_modes[:15], power[:15], color='steelblue')
            dom = int(np.argmax(power[1:13])) + 1
            ax.axvline(dom, color='red', linestyle='--', alpha=0.6,
                       label=f'n={dom}')
            ax.set_xlabel('mode n'); ax.set_ylabel('|F_n|^2')
            ax.set_title(f'Fourier: {title}')
            ax.set_xticks(n_modes[:15])
            ax.legend(); ax.grid(True, alpha=0.3)
            print(f"  {title}: dominant n = {dom}")

    plt.tight_layout()
    plt.savefig('figures/phi_polygon.png', dpi=150)
    plt.close()
    print("Saved figures/phi_polygon.png")


if __name__ == "__main__":
    main()
