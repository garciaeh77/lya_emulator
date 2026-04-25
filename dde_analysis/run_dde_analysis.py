"""
run_dde_analysis.py

Main analysis script. Evaluates:
1. Baseline chi-squared at best-fit parameters (LCDM emulator vs BOSS DR14)
2. Chi-squared with Union3 DDE template injected
3. Profile likelihood scan over template amplitude epsilon

Usage:
    python dde_analysis/run_dde_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.dde_likelihood import DDELikelihood

# ── Configuration ──────────────────────────────────────────────────────────
BASEDIR  = '/Users/helenagescu/lya_emulator/dtau-48-48'
TRAINDIR = '/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf'
MIN_Z    = 2.2
MAX_Z    = 4.6

# Best-fit parameter vector — set to None to use midpoint of ranges as test
PARAMS_BESTFIT = None

# Template amplitude grid for profile likelihood scan
EPSILON_GRID = np.linspace(0, 2, 21)

# Output directory
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTDIR, exist_ok=True)


# ── Step 1: Baseline likelihood ────────────────────────────────────────────
print("="*60)
print("Step 1: Setting up baseline LCDM likelihood")
print("="*60)

lik = LikelihoodClass(
    basedir=BASEDIR,
    mean_flux='s',
    min_z=MIN_Z,
    max_z=MAX_Z,
    optimise_GP=True,
    traindir=TRAINDIR,
    data_corr=True,
    sdss='dr14',
)

# Print parameter names
print("\nParameter vector:")
pnames = lik.get_pnames()
for i, (name, _) in enumerate(pnames):
    lo, hi = lik.param_limits[i]
    print(f"  [{i:2d}] {name:20s}  [{lo:.4f}, {hi:.4f}]")

# Set parameters
if PARAMS_BESTFIT is None:
    print("\nNo best-fit provided — using parameter range midpoints as test.")
    params = (lik.param_limits[:, 0] + lik.param_limits[:, 1]) / 2.
else:
    params = np.array(PARAMS_BESTFIT)

print(f"\nParameter vector:\n{params}")

# Evaluate baseline
ll_baseline = lik.likelihood(
    params, include_emu=True,
    data_power=None,
    hprior=False, oprior=False
)
chi2_baseline = -2 * ll_baseline
print(f"\nBaseline log-likelihood = {ll_baseline:.4f}")
print(f"Baseline chi2           = {chi2_baseline:.4f}")


# ── Step 2: DDE likelihood at epsilon=1 ───────────────────────────────────
print("\n" + "="*60)
print("Step 2: DDE likelihood at full Union3 amplitude (epsilon=1)")
print("="*60)

lik_dde = DDELikelihood(
    BASEDIR,
    mean_flux='s',
    min_z=MIN_Z,
    max_z=MAX_Z,
    optimise_GP=False,
    traindir=TRAINDIR,
    data_corr=True,
    sdss='dr14',
    epsilon=1.0,
)
lik_dde.gpemu    = lik.gpemu
lik_dde.icov_bin = lik.icov_bin
lik_dde.cdet     = lik.cdet

ll_dde     = lik_dde.likelihood(params, include_emu=True,
                                data_power=None,
                                hprior=False, oprior=False)
chi2_dde   = -2 * ll_dde
delta_chi2 = chi2_dde - chi2_baseline
significance = np.sqrt(np.abs(delta_chi2))
p_value = scipy.stats.chi2.sf(np.abs(delta_chi2), df=1)

print(f"\nDDE log-likelihood      = {ll_dde:.4f}")
print(f"DDE chi2                = {chi2_dde:.4f}")
print(f"\n{'='*60}")
print(f"Delta chi2              = {delta_chi2:+.4f}")
print(f"Significance (~1 dof)   = {significance:.2f} sigma")
print(f"p-value                 = {p_value:.4f}")
if delta_chi2 > 0:
    print("Result: BOSS data DISFAVOURS the Union3 DDE tilt")
else:
    print("Result: BOSS data PREFERS the Union3 DDE tilt")
print('='*60)


# ── Step 3: Profile likelihood scan over epsilon ───────────────────────────
print("\n" + "="*60)
print("Step 3: Profile likelihood scan over epsilon")
print("="*60)

delta_chi2_profile = []
for eps in EPSILON_GRID:
    lik_eps = DDELikelihood(
        BASEDIR,
        mean_flux='s',
        min_z=MIN_Z,
        max_z=MAX_Z,
        optimise_GP=False,
        traindir=TRAINDIR,
        data_corr=True,
        sdss='dr14',
        epsilon=eps,
    )
    lik_eps.gpemu    = lik.gpemu
    lik_eps.icov_bin = lik.icov_bin
    lik_eps.cdet     = lik.cdet

    ll = lik_eps.likelihood(params, include_emu=True,
                            data_power=None,
                            hprior=False, oprior=False)
    dchi2 = -2 * ll - chi2_baseline
    delta_chi2_profile.append(dchi2)
    print(f"  epsilon = {eps:.2f}   delta_chi2 = {dchi2:+.4f}")

delta_chi2_profile = np.array(delta_chi2_profile)

# Save results
outfile = os.path.join(OUTDIR, 'delta_chi2_profile_union3.txt')
np.savetxt(outfile,
           np.column_stack([EPSILON_GRID, delta_chi2_profile]),
           header='epsilon  delta_chi2',
           fmt='%.6f')
print(f"\nResults saved to {outfile}")

best_idx = np.argmin(delta_chi2_profile)
best_eps = EPSILON_GRID[best_idx]
print(f"Preferred epsilon = {best_eps:.2f}  "
      f"(delta_chi2 = {delta_chi2_profile[best_idx]:+.4f})")


# ── Step 4: Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(EPSILON_GRID, delta_chi2_profile,
        'o-', color='teal', linewidth=2, markersize=5)
ax.axhline(0,  color='k',     linestyle='--',
           linewidth=0.8, label='LCDM baseline')
ax.axhline(1,  color='gray',  linestyle=':',
           linewidth=0.8, label=r'1$\sigma$')
ax.axhline(4,  color='gray',  linestyle='-.',
           linewidth=0.8, label=r'2$\sigma$')
ax.axvline(1,  color='coral', linestyle='--',
           linewidth=1.2, label='Full DDE (Union3, $\\epsilon$=1)')
ax.axvline(best_eps, color='teal', linestyle=':',
           linewidth=1.0, label=f'Preferred $\\epsilon$={best_eps:.2f}')
ax.set_xlabel(r'Template amplitude $\epsilon$', fontsize=12)
ax.set_ylabel(r'$\Delta\chi^2$  (DDE $-$ baseline)', fontsize=12)
ax.set_title('Profile likelihood: Union3 DDE tilt vs BOSS DR14', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()

plotfile = os.path.join(OUTDIR, 'profile_likelihood_union3.png')
plt.savefig(plotfile, dpi=150)
print(f"Plot saved to {plotfile}")
plt.show()
