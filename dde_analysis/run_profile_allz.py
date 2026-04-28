"""
run_profile_allz.py
Profile likelihood scan over DDE template amplitude epsilon.
Uses ALL redshift bins (min_z=2.2).
Results saved to dde_analysis/results/all_zbins/
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.dde_likelihood import DDELikelihood

BASEDIR      = '/Users/helenagescu/lya_emulator/dtau-48-48'
TRAINDIR     = '/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf'
MIN_Z        = 2.2
MAX_Z        = 4.6
LABEL        = 'all_zbins'
EPSILON_GRID = np.linspace(0, 2, 21)
OUTDIR       = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', LABEL
)
os.makedirs(OUTDIR, exist_ok=True)

# ── Load best-fit params from chains ───────────────────────────────────────
params_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'bestfit_from_chains.txt'
)
if os.path.exists(params_file):
    params = np.loadtxt(params_file)
    print(f"Loaded best-fit params from chains: {params_file}")
else:
    print("WARNING: best-fit from chains not found. Using midpoint.")
    params = None

# ── Baseline likelihood ────────────────────────────────────────────────────
print("="*60)
print(f"Profile likelihood — {LABEL} (min_z={MIN_Z})")
print("="*60)

lik = LikelihoodClass(
    basedir=BASEDIR, mean_flux='s',
    min_z=MIN_Z, max_z=MAX_Z,
    optimise_GP=True, traindir=TRAINDIR,
    data_corr=True, sdss='dr14',
)

if params is None:
    params = (lik.param_limits[:, 0] + lik.param_limits[:, 1]) / 2.

print(f"Parameter vector: {params}")
print(f"Redshift bins: {lik.zout}")

ll_baseline  = lik.likelihood(params, include_emu=True,
                               data_power=None,
                               hprior=False, oprior=False)
chi2_baseline = -2 * ll_baseline
print(f"\nBaseline log-likelihood = {ll_baseline:.4f}")
print(f"Baseline chi2           = {chi2_baseline:.4f}")

# ── Profile scan ───────────────────────────────────────────────────────────
print(f"\nScanning epsilon from {EPSILON_GRID[0]} to {EPSILON_GRID[-1]}...")
delta_chi2_profile = []

for eps in EPSILON_GRID:
    lik_eps = DDELikelihood(
        BASEDIR, mean_flux='s',
        min_z=MIN_Z, max_z=MAX_Z,
        optimise_GP=False, traindir=TRAINDIR,
        data_corr=True, sdss='dr14',
        epsilon=eps,
    )
    lik_eps.gpemu    = lik.gpemu
    lik_eps.icov_bin = lik.icov_bin
    lik_eps.cdet     = lik.cdet

    ll   = lik_eps.likelihood(params, include_emu=True,
                               data_power=None,
                               hprior=False, oprior=False)
    dchi2 = -2 * ll - chi2_baseline
    delta_chi2_profile.append(dchi2)
    print(f"  epsilon = {eps:.2f}   delta_chi2 = {dchi2:+.4f}")

delta_chi2_profile = np.array(delta_chi2_profile)
best_idx = np.argmin(delta_chi2_profile)
best_eps = EPSILON_GRID[best_idx]

# ── Results summary ────────────────────────────────────────────────────────
dchi2_at_1    = np.interp(1.0, EPSILON_GRID, delta_chi2_profile)
significance  = np.sqrt(np.abs(dchi2_at_1))
p_value       = scipy.stats.chi2.sf(np.abs(dchi2_at_1), df=1)

print(f"\n{'='*60}")
print(f"Baseline chi2          = {chi2_baseline:.4f}")
print(f"Delta_chi2 at eps=1    = {dchi2_at_1:+.4f}")
print(f"Significance (~1 dof)  = {significance:.3f} sigma")
print(f"p-value                = {p_value:.4f}")
print(f"Preferred epsilon      = {best_eps:.2f}")
if dchi2_at_1 > 0:
    print("Result: BOSS data DISFAVOURS the Union3 DDE tilt")
else:
    print("Result: BOSS data PREFERS the Union3 DDE tilt")
print('='*60)

# ── Save results ──────────────────────────────────────────────────────────
outfile = os.path.join(OUTDIR, 'profile_likelihood.txt')
np.savetxt(outfile,
           np.column_stack([EPSILON_GRID, delta_chi2_profile]),
           header=f'Profile likelihood ({LABEL})\nepsilon  delta_chi2',
           fmt='%.6f')
print(f"\nResults saved to {outfile}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(EPSILON_GRID, delta_chi2_profile,
        'o-', color='teal', linewidth=2, markersize=5,
        label=f'{LABEL}')
ax.axhline(0, color='k',    linestyle='--',
           linewidth=0.8, label='LCDM baseline')
ax.axhline(1, color='gray', linestyle=':',
           linewidth=0.8, label=r'1$\sigma$')
ax.axhline(4, color='gray', linestyle='-.',
           linewidth=0.8, label=r'2$\sigma$')
ax.axvline(1, color='coral', linestyle='--',
           linewidth=1.2, label='Full DDE (Union3, $\\epsilon$=1)')
ax.axvline(best_eps, color='teal', linestyle=':',
           linewidth=1.0,
           label=f'Preferred $\\epsilon$={best_eps:.2f}')
ax.set_xlabel(r'Template amplitude $\epsilon$', fontsize=12)
ax.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax.set_title(
    f'Profile likelihood — {LABEL}\n'
    f'Union3 DDE tilt vs BOSS DR14',
    fontsize=11
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()

plotfile = os.path.join(OUTDIR, 'profile_likelihood.png')
plt.savefig(plotfile, dpi=150)
print(f"Plot saved to {plotfile}")
