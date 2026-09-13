"""
core.py

Reaction kinetics and time-stepping routines for the
Liesegang hypergraph project.

Reaction-diffusion system:
    da/dt = D_a * Laplacian(a) - R
    db/dt = D_b * Laplacian(b) - R
    dc/dt = R

Ostwald-Prager kinetics with saturation:
    R = k_nuc  * max(0, a*b - K_nuc) * (1 - c/c_max)
      + k_auto * c * max(0, a*b - K_sp)  * (1 - c/c_max)
"""

import numpy as np


DEFAULTS = dict(
    Da=1.0,
    Db=0.01,
    a_left=1.0,
    b0=1.0,
    K_nuc=0.55,
    K_sp=0.10,
    k_nuc=0.5,
    k_auto=20.0,
    c_max=1.0,
)


def reaction_rate(a, b, c,
                  K_nuc=0.55, K_sp=0.10,
                  k_nuc=0.5, k_auto=20.0, c_max=1.0):
    """Ostwald-Prager kinetics with saturation."""
    sup = np.clip(a * b, 0.0, 10.0)
    sat = np.clip(1.0 - c / c_max, 0.0, 1.0)
    return (k_nuc  * np.maximum(0.0, sup - K_nuc) * sat
          + k_auto * c * np.maximum(0.0, sup - K_sp) * sat)


def lap_1d(u, dx):
    """1D Laplacian, no-flux boundaries (2nd order)."""
    L = np.zeros_like(u)
    L[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
    L[0]    = 2.0 * (u[1]  - u[0])  / dx**2
    L[-1]   = 2.0 * (u[-2] - u[-1]) / dx**2
    return L


def lap_2d_cart(u, dx, dy):
    """2D Cartesian Laplacian, no-flux via edge padding."""
    u_pad = np.pad(u, 1, mode='edge')
    return ((u_pad[2:, 1:-1] - 2.0 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx**2
          + (u_pad[1:-1, 2:] - 2.0 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy**2)


def lap_2d_polar(u, r, dr, dtheta):
    """
    Laplacian in polar coordinates:
        Lap u = d_rr u + (1/r) d_r u + (1/r^2) d_theta_theta u

    Periodic in theta, mirror/no-flux in r.
    u shape: (Nr, Ntheta).
    """
    Nr, Nt = u.shape
    inv_r = np.zeros(Nr); inv_r[1:] = 1.0 / r[1:]
    inv_r2 = np.zeros(Nr); inv_r2[1:] = 1.0 / r[1:]**2

    u_r_pad = np.pad(u, ((1, 1), (0, 0)), mode='edge')
    u_t_pad = np.pad(u, ((0, 0), (1, 1)), mode='wrap')

    la_rr = (u_r_pad[2:, :] - 2.0 * u + u_r_pad[:-2, :]) / dr**2
    la_r  = inv_r[:, None] * (u_r_pad[2:, :] - u_r_pad[:-2, :]) / (2.0 * dr)
    la_tt = inv_r2[:, None] * (u_t_pad[:, 2:] - 2.0 * u + u_t_pad[:, :-2]) / dtheta**2

    return la_rr + la_r + la_tt


def stable_dt(params, h_min=0.1):
    """
    Stability limit for explicit Euler scheme.

    dt <= min( dt_diff , dt_rxn )
    """
    dt_diff = 0.4 * h_min**2 / max(params['Da'], params['Db'], 1e-9)
    dt_rxn  = 0.05 / max(params['k_auto'] * params['a_left'],
                         params['k_nuc'], 1e-9)
    return min(dt_diff, dt_rxn)


def euler_step_1d(a, b, c, dt, dx, p):
    """One Euler step for 1D system."""
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
    return a, b, c


def euler_step_2d_cart(a, b, c, dt, dx, dy, p, src_mask):
    """One Euler step for 2D Cartesian system."""
    R = reaction_rate(a, b, c,
                      p['K_nuc'], p['K_sp'],
                      p['k_nuc'], p['k_auto'], p['c_max'])
    a += dt * (p['Da'] * lap_2d_cart(a, dx, dy) - R)
    b += dt * (p['Db'] * lap_2d_cart(b, dx, dy) - R)
    c += dt * R
    a[src_mask] = p['a_left']
    np.clip(a, 0.0, None, out=a)
    np.clip(b, 0.0, None, out=b)
    np.clip(c, 0.0, p['c_max'], out=c)
    return a, b, c


def count_bands_1d(c, c_min=0.3, min_gap=3):
    """
    Count bands as connected components of c > c_min.
    Returns list of band-center indices.
    """
    mask = c > c_min
    bands = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            bands.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    if len(bands) > 1:
        filt = [bands[0]]
        for bb in bands[1:]:
            if bb - filt[-1] >= min_gap:
                filt.append(bb)
        bands = filt
    return bands


def count_rings(c, c_min=0.3, min_gap=4):
    """Same as count_bands_1d, but named for radial geometry."""
    return count_bands_1d(c, c_min=c_min, min_gap=min_gap)
