"""
desi_dr1_analysis.py

Fisher forecast for detecting the Union3 DDE spectral tilt using
compressed Lyman-alpha P1D likelihood constraints from multiple surveys.

The compressed parameters are:
  - Delta2_star: dimensionless amplitude of linear P(k) at z=3, k=0.009 km/s
  - n_star:      local spectral slope at the same pivot point

Datasets included:
  - SDSS DR2     (McDonald et al. 2006)
  - SDSS DR9     (Palanque-Delabrouille et al. 2015)
  - SDSS DR14a   (Chabanier et al. 2019)
  - SDSS DR14b   (Walther et al. 2024)
  - DESI DR1     (Chaves-Montero et al. 2026)

Usage:
    python dde_analysis/desi_dr1_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dde_analysis.template import make_template_interpolator

# ── All datasets from yaml files ───────────────────────────────────────────
DATASETS = {
    'SDSS DR2\n(McDonald+2006)': {
        'delta2star_mean': 0.47,
        'delta2star_std':  0.06,
        'nstar_mean':      -2.300,
        'nstar_std':       0.055,
        'correlation':     0.60,
        'zstar':           3.0,
        'kstar_kms':       0.009,
        'color':           '#8B4513',
        'year':            2006,
    },
    'SDSS DR9\n(PD+2015)': {
        'delta2star_mean': 0.32,
        'delta2star_std':  0.03,
        'nstar_mean':      -2.360,
        'nstar_std':       0.010,
        'correlation':     0.55,
        'zstar':           3.0,
        'kstar_kms':       0.009,
        'color':           '#E07B39',
        'year':            2015,
    },
    'SDSS DR14a\n(Chabanier+2019)': {
        'delta2star_mean': 0.310,
        'delta2star_std':  0.020,
        'nstar_mean':      -2.340,
        'nstar_std':       0.006,
        'correlation':     0.512,
        'zstar':           3.0,
        'kstar_kms':       0.009,
        'color':           '#4A90D9',
        'year':            2019,
    },
    'SDSS DR14b\n(Walther+2024)': {
        'delta2star_mean': 0.388,
        'delta2star_std':  0.045,
        'nstar_mean':      -2.2978,
        'nstar_std':       0.0067,
        'correlation':     0.632,
        'zstar':           3.0,
        'kstar_kms':       0.009,
        'color':           '#7B68EE',
        'year':            2024,
    },
    'DESI DR1\n(Chaves-Montero+2026)': {
        'delta2star_mean': 0.379,
        'delta2star_std':  0.032,
        'nstar_mean':      -2.309,
        'nstar_std':       0.019,
        'correlation':     -0.1738,
        'zstar':           3.0,
        'kstar_kms':       0.009,
        'color':           'teal',
        'year':            2026,
    },
}

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'desi_dr1'
)
os.makedirs(OUTDIR, exist_ok=True)

T_fn = make_template_interpolator()


# ── DDE shift calculation ──────────────────────────────────────────────────
def compute_dde_shifts(zstar, kstar_kms, epsilon=1.0, dk_frac=0.1):
    """
    Compute DDE-induced shifts in (Delta2_star, n_star) at pivot point.

    Delta2_star shift: fractional change in flux power at pivot scale,
    approximating the change in matter power amplitude through Lya bias.

    n_star shift: change in spectral slope from the scale-dependent tilt,
    computed as d log T / d log k at the pivot scale.
    """
    T_pivot = T_fn(np.array([kstar_kms]), zstar, epsilon=epsilon)[0]
    delta_Delta2_star_frac = T_pivot - 1.0

    k_hi = kstar_kms * (1 + dk_frac)
    k_lo = kstar_kms * (1 - dk_frac)
    T_hi = T_fn(np.array([k_hi]), zstar, epsilon=epsilon)[0]
    T_lo = T_fn(np.array([k_lo]), zstar, epsilon=epsilon)[0]

    # Logarithmic slope of template = change in spectral index
    if T_hi > 0 and T_lo > 0:
        delta_n_star = (np.log(T_hi) - np.log(T_lo)) / \
                       (np.log(k_hi) - np.log(k_lo))
    else:
        delta_n_star = 0.0

    return delta_Delta2_star_frac, delta_n_star


def fisher_sn(dataset, epsilon=1.0):
    """
    Compute Fisher S/N for detecting DDE shift using 2x2 covariance.
    Returns S/N, delta_theta vector, and 2x2 covariance matrix.
    """
    zstar     = dataset['zstar']
    kstar_kms = dataset['kstar_kms']
    sig1      = dataset['delta2star_std']
    sig2      = dataset['nstar_std']
    rho       = dataset['correlation']

    C = np.array([
        [sig1**2,       rho*sig1*sig2],
        [rho*sig1*sig2, sig2**2      ],
    ])
    C_inv = np.linalg.inv(C)

    frac_shift, slope_shift = compute_dde_shifts(
        zstar, kstar_kms, epsilon=epsilon
    )

    delta_Delta2 = dataset['delta2star_mean'] * frac_shift
    delta_theta  = np.array([delta_Delta2, slope_shift])
    SN_sq        = delta_theta @ C_inv @ delta_theta
    SN           = np.sqrt(SN_sq)

    return SN, delta_theta, C


# ── Print DDE shifts at pivot ──────────────────────────────────────────────
print("="*65)
print("DDE template shifts at pivot point (z=3.0, k=0.009 km/s)")
print("="*65)
frac, slope = compute_dde_shifts(3.0, 0.009, epsilon=1.0)
print(f"Union3 DDE (epsilon=1):")
print(f"  Fractional Delta2_star shift: {frac:+.6f}  ({frac*100:+.4f}%)")
print(f"  Additive n_star shift:        {slope:+.6f}")


# ── Fisher S/N for all datasets ────────────────────────────────────────────
print("\n" + "="*65)
print("Fisher S/N for detecting Union3 DDE shift — all datasets")
print("="*65)
print(f"\n{'Dataset':35s}  {'sig_D2*':>8s}  {'sig_n*':>7s}  "
      f"{'S/N':>7s}  {'Need Nx':>8s}")
print("-"*75)

results = {}
for name, ds in DATASETS.items():
    SN, delta_theta, C = fisher_sn(ds, epsilon=1.0)
    N_needed = (2.0/SN)**2 if SN > 0 else np.inf
    results[name] = {
        'SN': SN, 'delta_theta': delta_theta,
        'C': C, 'N_needed': N_needed
    }
    label = name.replace('\n', ' ')
    print(f"{label:35s}  {ds['delta2star_std']:>8.4f}  "
          f"{ds['nstar_std']:>7.4f}  {SN:>7.4f}  {N_needed:>8.1f}x")


# ── Epsilon scan ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("S/N vs epsilon for all datasets")
print("="*65)
eps_grid   = np.linspace(0, 2, 41)
sn_vs_eps  = {}
for name, ds in DATASETS.items():
    sns = [fisher_sn(ds, epsilon=e)[0] for e in eps_grid]
    sn_vs_eps[name] = np.array(sns)

# Volume needed for each dataset
print(f"\n{'Dataset':35s}  {'S/N(eps=1)':>10s}  {'Need Nx':>10s}")
print("-"*60)
for name, ds in DATASETS.items():
    sn1 = results[name]['SN']
    N   = results[name]['N_needed']
    label = name.replace('\n', ' ')
    print(f"{label:35s}  {sn1:>10.4f}  {N:>10.1f}x")

# Comparison with full P1D Fisher result
print("\n" + "="*65)
print("Comparison: compressed vs full P1D Fisher sensitivity")
print("="*65)
print(f"  BOSS DR14 full P1D (all-z):          S/N = 0.2287")
print(f"  SDSS DR14a compressed (Chabanier):   "
      f"S/N = {results['SDSS DR14a\n(Chabanier+2019)']['SN']:.4f}")
print(f"  DESI DR1 compressed (Chaves-Montero): "
      f"S/N = {results['DESI DR1\n(Chaves-Montero+2026)']['SN']:.4f}")
print()
print("Note: full P1D uses all k modes and redshift bins.")
print("Compressed uses only 2 numbers at a single pivot point.")
print("Full P1D should in principle be more sensitive.")


# ── Save results ───────────────────────────────────────────────────────────
outfile = os.path.join(OUTDIR, 'compressed_p1d_summary.txt')
with open(outfile, 'w') as f:
    f.write("DDE Fisher forecast — compressed P1D likelihood\n")
    f.write("="*60 + "\n\n")
    f.write(f"Pivot: z_star=3.0, k_star=0.009 km/s\n")
    f.write(f"Union3 DDE shifts at pivot:\n")
    frac, slope = compute_dde_shifts(3.0, 0.009, epsilon=1.0)
    f.write(f"  Delta2_star fractional shift: {frac:+.6f}\n")
    f.write(f"  n_star additive shift:        {slope:+.6f}\n\n")
    for name, res in results.items():
        label = name.replace('\n', ' ')
        f.write(f"{label}:\n")
        f.write(f"  S/N      = {res['SN']:.6f}\n")
        f.write(f"  Need Nx  = {res['N_needed']:.1f}x\n\n")
print(f"\nSummary saved to {outfile}")


# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Plot 1: S/N vs epsilon ─────────────────────────────────────────────────
ax = axes[0]
for name, sns in sn_vs_eps.items():
    ds = DATASETS[name]
    label = name.replace('\n', ' ')
    ax.plot(eps_grid, sns, color=ds['color'],
            linewidth=2, label=label)
ax.axhline(2, color='red',   linestyle='--', linewidth=1.5,
           label='2σ threshold')
ax.axhline(1, color='gray',  linestyle=':',  linewidth=0.8)
ax.axvline(1, color='black', linestyle='--', linewidth=0.8,
           alpha=0.5, label='Full Union3 DDE')
ax.set_xlabel(r'Template amplitude $\epsilon$', fontsize=12)
ax.set_ylabel('S/N', fontsize=12)
ax.set_title('Compressed P1D: S/N vs template amplitude\nUnion3 DDE', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)


# ── Plot 2: S/N vs data volume ─────────────────────────────────────────────
ax = axes[1]
vols = np.logspace(0, 4, 200)
for name, res in results.items():
    ds    = DATASETS[name]
    label = name.replace('\n', ' ')
    ax.loglog(vols, res['SN'] * np.sqrt(vols),
              color=ds['color'], linewidth=2, label=label)
ax.axhline(2, color='red',  linestyle='--', linewidth=1.5,
           label='2σ threshold')
ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8)
# Mark survey volumes relative to SDSS DR2
for label_s, vf, col in [
    ('SDSS DR2',    1,    '#8B4513'),
    ('SDSS DR9',    3,    '#E07B39'),
    ('SDSS DR14',   10,   '#4A90D9'),
    ('DESI DR1',    17,   'teal'),
    ('DESI full',   100,  'navy'),
]:
    ax.axvline(vf, color=col, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.text(vf*1.1, 0.05, label_s, fontsize=6,
            color=col, rotation=90, va='bottom')
ax.set_xlabel('Data volume relative to SDSS DR2', fontsize=12)
ax.set_ylabel('S/N', fontsize=12)
ax.set_title('Compressed P1D: S/N vs data volume\nUnion3 DDE', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, which='both')


# ── Plot 3: Error ellipses in (Delta2_star, n_star) space ─────────────────
ax = axes[2]
for name, ds in DATASETS.items():
    label = name.replace('\n', ' ')
    sig1  = ds['delta2star_std']
    sig2  = ds['nstar_std']
    rho   = ds['correlation']
    C_ds  = np.array([[sig1**2,       rho*sig1*sig2],
                       [rho*sig1*sig2, sig2**2      ]])
    vals, vecs = np.linalg.eigh(C_ds)
    angle = np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))

    for n_sig, alpha in [(1, 0.35), (2, 0.15)]:
        w = 2 * n_sig * np.sqrt(vals[1])
        h = 2 * n_sig * np.sqrt(vals[0])
        ell = Ellipse(
            xy=(ds['delta2star_mean'], ds['nstar_mean']),
            width=w, height=h, angle=angle,
            color=ds['color'], alpha=alpha
        )
        ax.add_patch(ell)

    ax.plot(ds['delta2star_mean'], ds['nstar_mean'],
            'o', color=ds['color'], markersize=6, label=label)

    # Arrow showing DDE shift at epsilon=1
    frac, slope = compute_dde_shifts(
        ds['zstar'], ds['kstar_kms'], epsilon=1.0
    )
    ddelta = ds['delta2star_mean'] * frac
    ax.annotate('',
                xy=(ds['delta2star_mean'] + ddelta * 50,
                    ds['nstar_mean']      + slope  * 50),
                xytext=(ds['delta2star_mean'], ds['nstar_mean']),
                arrowprops=dict(arrowstyle='->', color='coral', lw=1.5))

ax.set_xlabel(r'$\Delta^2_*$', fontsize=12)
ax.set_ylabel(r'$n_*$', fontsize=12)
ax.set_title('Compressed P1D constraints\n'
             '(arrow = 50x Union3 DDE shift)', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)
ax.autoscale()

plt.suptitle('DDE template analysis — compressed Ly\u03b1 P1D likelihood',
             fontsize=13)
plt.tight_layout()
plotfile = os.path.join(OUTDIR, 'compressed_p1d_dde_analysis.png')
plt.savefig(plotfile, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plotfile}")
print("\nDONE")
