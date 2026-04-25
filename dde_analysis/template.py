"""
template.py

Loads the digitised DDE spectral tilt template from Figure 9 of
Garza et al. 2026 (arXiv:2601.00767) for the Union3 DDE model,
and provides an interpolation function T(k, z, epsilon) for use
in the likelihood injection.

The template encodes the ratio:
    [Delta^2_DDE(k,z) / Delta^2_LCDM(k,z)] - 1
as digitised from Figure 9. The multiplicative template is then:
    T(k, z, epsilon) = 1 + epsilon * ratio(k, z)
where epsilon=0 is LCDM and epsilon=1 is the full Union3 DDE prediction.
"""

import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# ── Redshift snapshots matching Figure 9 of Garza et al. ─────────────────
REDSHIFTS = np.array([
    1.317, 1.491, 2.2, 2.33, 2.4, 2.6, 2.8,
    3.0, 3.2, 3.4, 3.6, 4.0
])

Z_TO_FNAME = {
    4.0:   'z4p0',
    3.6:   'z3p6',
    3.4:   'z3p4',
    3.2:   'z3p2',
    3.0:   'z3p0',
    2.8:   'z2p8',
    2.6:   'z2p6',
    2.4:   'z2p4',
    2.33:  'z2p33',
    2.2:   'z2p2',
    1.491: 'z1p491',
    1.317: 'z1p317',
}

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'templates', 'union3'
)


def _load_single_csv(z):
    """
    Load digitised CSV for a single redshift.
    Returns k_array (s/km) and ratio_array, sorted by k.
    """
    fname = os.path.join(TEMPLATE_DIR, f"union3_{Z_TO_FNAME[z]}.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Template file not found: {fname}\n"
            f"Expected files in: {TEMPLATE_DIR}"
        )
    # Load — no header row in your CSVs
    data = np.loadtxt(fname, delimiter=',')

    # Sort by k
    idx = np.argsort(data[:, 0])
    k     = data[idx, 0]
    ratio = data[idx, 1]

    return k, ratio


def load_all_templates():
    """
    Load all 12 redshift CSVs and interpolate onto a common k grid.
    Returns:
        k_grid     : 1D array, shape (n_k,), k in s/km
        ratio_grid : 2D array, shape (n_z, n_k), rows = ascending redshift
    """
    k_all, ratio_all = [], []
    for z in np.sort(REDSHIFTS):
        k, ratio = _load_single_csv(z)
        k_all.append(k)
        ratio_all.append(ratio)

    # Common k range across all redshift bins
    k_min = max(k[0]  for k in k_all)
    k_max = min(k[-1] for k in k_all)
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), 100)

    # Interpolate each bin onto common k grid
    ratio_grid = np.zeros((len(REDSHIFTS), len(k_grid)))
    for i, (k_z, r_z) in enumerate(zip(k_all, ratio_all)):
        ratio_grid[i] = np.interp(k_grid, k_z, r_z)

    return k_grid, ratio_grid


def make_template_interpolator():
    """
    Build and return a 2D interpolator over (z, log10 k).

    Returns a callable T(k_kms, z, epsilon=1.0).
    """
    k_grid, ratio_grid = load_all_templates()
    log_k_grid = np.log10(k_grid)
    z_grid = np.sort(REDSHIFTS)

    interp = RegularGridInterpolator(
        (z_grid, log_k_grid),
        ratio_grid,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    def template(k_kms, z, epsilon=1.0):
        k_kms = np.asarray(k_kms)
        log_k = np.log10(k_kms)
        points = np.column_stack([
            np.full_like(log_k, z),
            log_k
        ])
        ratio = interp(points)
        return 1.0 + epsilon * ratio

    return template


def plot_template(save=True):
    """
    Validation plot: reproduce Figure 9 of Garza et al.
    Compare visually against the paper before proceeding.
    """
    k_grid, ratio_grid = load_all_templates()
    z_sorted = np.sort(REDSHIFTS)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, z in enumerate(z_sorted):
        ax = axes[i]
        ax.semilogx(k_grid, ratio_grid[i] * 100,
                    color='goldenrod', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
        ax.set_title(f'z = {z}', fontsize=10)
        ax.set_xlabel('k [s/km]', fontsize=8)
        ax.set_ylabel(r'$[\Delta^2_{DDE}/\Delta^2_\Lambda - 1]$ [%]',
                      fontsize=7)
        ax.set_ylim(-8, 4)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Union3 DDE template — digitised from Garza et al. 2026 Fig. 9',
        fontsize=13
    )
    plt.tight_layout()

    if save:
        outpath = os.path.join(TEMPLATE_DIR,
                               'template_validation_union3.png')
        plt.savefig(outpath, dpi=150)
        print(f"Saved validation plot to:\n  {outpath}")

    plt.show()


if __name__ == '__main__':
    print("Checking template files in:")
    print(f"  {TEMPLATE_DIR}\n")

    # Check all files exist
    all_ok = True
    for z in np.sort(REDSHIFTS):
        fname = f"union3_{Z_TO_FNAME[z]}.csv"
        fpath = os.path.join(TEMPLATE_DIR, fname)
        status = "OK" if os.path.exists(fpath) else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  {fname:30s}  {status}")

    if not all_ok:
        print("\nSome files are missing — fix before proceeding.")
        exit(1)

    print("\nAll files found. Loading and plotting...")
    plot_template(save=True)

    print("\nTesting interpolator at a few points...")
    T = make_template_interpolator()
    k_test = np.array([0.003, 0.01, 0.03, 0.1])
    for z_test in [4.0, 3.0, 2.2, 1.317]:
        vals = T(k_test, z_test, epsilon=1.0)
        print(f"  z={z_test:<6}  T(k) = {np.round(vals, 4)}")

    print("\nDone — check the plot against Figure 9 of the paper.")
