"""
make_all_figures.py

Generate all figures for the paper in one run.

Run:  python make_all_figures.py
Output: all figures in figures/ directory.
"""

import os
import time
import subprocess
import sys


MODULES = [
    ("base_1d",          "BASE 1D: parallel bands"),
    ("base_2d_polar",    "BASE 2D radial: concentric rings"),
    ("base_2d_cart",     "BASE 2D Cartesian"),
    ("ops.branch",       "Phi_branch: C-shaped fracture"),
    ("ops.radial",       "Phi_radial: B gradient from wall"),
    ("ops.ads",          "Phi_ads: cumulative adsorption"),
    ("ops.field",        "Phi_field: external drift"),
    ("ops.curv",         "Phi_curv: Liesegang eyes"),
    ("ops.symbreak",     "Phi_symbreak: speckled patterns"),
    ("ops.surf",         "Phi_surf: dendrites"),
    ("ops.interf",       "Phi_interf: anastomoses"),
    ("ops.secondary",    "Phi_secondary: secondary banding"),
    ("ops.dissolve",     "Phi_dissolve: precipitation waves"),
    ("ops.redox",        "Phi_redox: redox dissolution"),
    ("ops.compete",      "Phi_compete: two-precipitate dynamics"),
    ("ops.polygon",      "Phi_polygon: dislocations"),
    ("ops.ablation",     "Ablation protocol"),
]


def main():
    os.makedirs('figures', exist_ok=True)
    print("=" * 60)
    print("Generating all figures for the paper")
    print("=" * 60)
    t0 = time.time()

    for i, (mod, desc) in enumerate(MODULES, 1):
        print(f"\n[{i:2d}/{len(MODULES)}] {desc}")
        print("-" * 60)
        t_start = time.time()
        try:
            subprocess.run([sys.executable, "-m", mod], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR in {mod}: {e}")
        t_el = time.time() - t_start
        print(f"  -> done in {t_el:.1f} s")

    total = time.time() - t0
    print("\n" + "=" * 60)
    print(f"All figures generated in {total:.1f} s")
    print(f"Output directory: figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
