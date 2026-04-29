"""
run_all_configurations.py

Runs both Fisher forecast and profile likelihood scan for three
redshift configurations:
  1. all_z:           min_z = 2.2 (all redshift bins)
  2. no_z2p2:         min_z = 2.4 (exclude z=2.2)
  3. no_z2p2_no_z2p4: min_z = 2.6 (exclude z=2.2 and z=2.4)

Results saved to dde_analysis/results/{label}/
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats
import os, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.dde_likelihood import DDELikelihood
from dde_analysis.template import make_template_interpolator

# ── Configuration ──────────────────────────────────────────────────────────
BASEDIR      = '/Users/helenagescu/lya_emulator/dtau-48-48'
TRAINDIR     = '/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf'
MAX_Z        = 4.6
EPSILON_GRID = np.linspace(0, 2, 21)

RESULTS_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results'
)

CONFIGS = [
    {
        'label':       'all_z',
        'min_z':       2.2,
        'params_file': 'bestfit_z2.2-4.6.txt',
        'color':       'teal',
        'description': 'All redshift bins (z=2.2 to 4.6)',
    },
    {
        'label':       'no_z2p2',
        'min_z':       2.4,
        'params_file': 'bestfit_z2.6-4.6.txt',
        'color':       'steelblue',
        'description': 'Excluding z=2.2 (z=2.4 to 4.6)',
    },
    {
        'label':       'no_z2p2_no_z2p4',
        'min_z':       2.6,
        'params_file': 'bestfit_z2.6-4.6.txt',
        'color':       'coral',
        'description': 'Excluding z=2.2 and z=2.4 (z=2.6 to 4.6)',
    },
]

T_fn = make_template_interpolator()

# Store results for comparison plot
all_fisher  = {}
all_profile = {}

# ══════════════════════════════════════════════════════════════════════════
# LOOP OVER CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════

for cfg in CONFIGS:
    label       = cfg['label']
    min_z       = cfg['min_z']
    params_file = cfg['params_file']
    color       = cfg['color']
    desc        = cfg['description']

    OUTDIR = os.path.join(RESULTS_BASE, label)
    os.makedirs(OUTDIR, exist_ok=True)

    print("\n" + "="*70)
    print(f"CONFIGURATION: {label}")
    print(f"  {desc}")
    print(f"  min_z = {min_z}   params = {params_file}")
    print("="*70)

    # ── Load likelihood ────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lik = LikelihoodClass(
            basedir=BASEDIR, mean_flux='s',
            min_z=min_z, max_z=MAX_Z,
            optimise_GP=True, traindir=TRAINDIR,
            data_corr=True, sdss='dr14',
        )

    print(f"Redshift bins: {lik.zout}")
    print(f"k modes: {len(lik.kf)}   Total data points: {len(lik.zout)*len(lik.kf)}")

    # ── Load best-fit parameters ───────────────────────────────────────────
    pfile = os.path.join(RESULTS_BASE, params_file)
    if os.path.exists(pfile):
        params = np.loadtxt(pfile)
        print(f"Loaded params from {params_file}")
    else:
        print(f"WARNING: {params_file} not found, using midpoint")
        params = (lik.param_limits[:,0] + lik.param_limits[:,1]) / 2.

    # ── Evaluate baseline chi-squared ──────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll_baseline = lik.likelihood(
            params, include_emu=True,
            data_power=None, hprior=False, oprior=False
        )
    chi2_baseline = -2 * ll_baseline
    print(f"Baseline chi2 = {chi2_baseline:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # FISHER FORECAST
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Fisher forecast ({label}) ---")

    template_signal = []
    for bb, z in enumerate(lik.zout):
        T = T_fn(lik.kf, z, epsilon=1.0)
        template_signal.append(T - 1.0)
    template_signal = np.concatenate(template_signal)

    cov_blocks = [lik.get_BOSS_error(bb) for bb in range(len(lik.zout))]
    C_full     = scipy.linalg.block_diag(*cov_blocks)
    C_inv      = np.linalg.inv(C_full)

    SN_sq = template_signal @ C_inv @ template_signal
    SN    = np.sqrt(SN_sq)

    sn_desi_dr2  = SN * np.sqrt(2)
    sn_desi_full = SN * np.sqrt(10)
    f_needed     = (SN / 2.0)**2
    N_needed     = 1.0 / f_needed if f_needed < 1 else None

    # Per-bin S/N
    n_k = len(lik.kf)
    sn_per_bin = []
    for bb, z in enumerate(lik.zout):
        T_bb  = template_signal[bb*n_k:(bb+1)*n_k]
        ic_bb = np.linalg.inv(cov_blocks[bb])
        sn_per_bin.append(np.sqrt(T_bb @ ic_bb @ T_bb))
    sn_per_bin = np.array(sn_per_bin)

    print(f"  S/N BOSS DR14:    {SN:.4f}")
    print(f"  S/N DESI DR2:     {sn_desi_dr2:.4f}")
    print(f"  S/N DESI full:    {sn_desi_full:.4f}")
    if N_needed:
        print(f"  Need {N_needed:.1f}x BOSS for 2-sigma")
    print(f"  Per-bin S/N:")
    for z, sn in zip(lik.zout, sn_per_bin):
        print(f"    z={z:.2f}  S/N={sn:.4f}")

    all_fisher[label] = {
        'SN': SN, 'sn_desi_dr2': sn_desi_dr2,
        'sn_desi_full': sn_desi_full, 'N_needed': N_needed,
        'sn_per_bin': sn_per_bin, 'zout': lik.zout.copy(),
        'color': color, 'desc': desc,
    }

    # Save Fisher summary
    fisher_file = os.path.join(OUTDIR, 'fisher_summary.txt')
    with open(fisher_file, 'w') as f:
        f.write(f"Fisher forecast — {label}\n{'='*50}\n")
        f.write(f"min_z = {min_z}   n_zbins = {len(lik.zout)}\n")
        f.write(f"Total data points: {len(lik.zout)*len(lik.kf)}\n\n")
        f.write(f"S/N BOSS DR14:          {SN:.6f}\n")
        f.write(f"S/N DESI DR2 (2x):      {sn_desi_dr2:.6f}\n")
        f.write(f"S/N DESI full (10x):    {sn_desi_full:.6f}\n")
        if N_needed:
            f.write(f"Data volume for 2-sigma: {N_needed:.1f}x BOSS DR14\n")
        f.write("\nPer-redshift-bin S/N:\n")
        for z, sn in zip(lik.zout, sn_per_bin):
            f.write(f"  z={z:.3f}   S/N={sn:.6f}\n")
    print(f"  Fisher summary saved to {fisher_file}")

    # ══════════════════════════════════════════════════════════════════════
    # PROFILE LIKELIHOOD
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Profile likelihood ({label}) ---")

    delta_chi2_profile = []
    for eps in EPSILON_GRID:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lik_eps = DDELikelihood(
                BASEDIR, mean_flux='s',
                min_z=min_z, max_z=MAX_Z,
                optimise_GP=False, traindir=TRAINDIR,
                data_corr=True, sdss='dr14',
                epsilon=eps,
            )
        lik_eps.gpemu    = lik.gpemu
        lik_eps.icov_bin = lik.icov_bin
        lik_eps.cdet     = lik.cdet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ll   = lik_eps.likelihood(params, include_emu=True,
                                      data_power=None,
                                      hprior=False, oprior=False)
        dchi2 = -2 * ll - chi2_baseline
        delta_chi2_profile.append(dchi2)
        print(f"  epsilon={eps:.2f}  delta_chi2={dchi2:+.4f}")

    delta_chi2_profile = np.array(delta_chi2_profile)
    best_eps   = EPSILON_GRID[np.argmin(delta_chi2_profile)]
    dchi2_at_1 = float(np.interp(1.0, EPSILON_GRID, delta_chi2_profile))
    sig        = np.sqrt(np.abs(dchi2_at_1))
    pval       = scipy.stats.chi2.sf(np.abs(dchi2_at_1), df=1)

    print(f"\n  Baseline chi2         = {chi2_baseline:.4f}")
    print(f"  Delta_chi2 at eps=1   = {dchi2_at_1:+.4f}")
    print(f"  Significance          = {sig:.3f} sigma")
    print(f"  p-value               = {pval:.6f}")
    print(f"  Preferred epsilon     = {best_eps:.2f}")
    if dchi2_at_1 > 0:
        print(f"  Result: BOSS data DISFAVOURS the Union3 DDE tilt")
    else:
        print(f"  Result: BOSS data PREFERS the Union3 DDE tilt")

    all_profile[label] = {
        'delta_chi2': delta_chi2_profile,
        'chi2_baseline': chi2_baseline,
        'dchi2_at_1': dchi2_at_1,
        'significance': sig,
        'pval': pval,
        'best_eps': best_eps,
        'color': color,
        'desc': desc,
    }

    # Save profile results
    profile_file = os.path.join(OUTDIR, 'profile_likelihood.txt')
    np.savetxt(profile_file,
               np.column_stack([EPSILON_GRID, delta_chi2_profile]),
               header=f'Profile likelihood ({label})\nepsilon  delta_chi2',
               fmt='%.6f')
    print(f"  Profile saved to {profile_file}")

    # Individual profile plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(EPSILON_GRID, delta_chi2_profile, 'o-',
            color=color, linewidth=2, markersize=5)
    ax.axhline(0, color='k',    linestyle='--', linewidth=0.8, label='LCDM')
    ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8, label=r'1$\sigma$')
    ax.axhline(4, color='gray', linestyle='-.', linewidth=0.8, label=r'2$\sigma$')
    ax.axvline(1, color='coral',linestyle='--', linewidth=1.2,
               label='Full DDE ($\\epsilon$=1)')
    ax.set_xlabel(r'Template amplitude $\epsilon$', fontsize=12)
    ax.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
    ax.set_title(f'Profile likelihood — {label}\n{desc}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'profile_likelihood.png'), dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPARISON PLOTS")
print("="*70)

# ── Plot 1: Profile likelihood comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for label, res in all_profile.items():
    ax.plot(EPSILON_GRID, res['delta_chi2'],
            'o-', color=res['color'], linewidth=2, markersize=4,
            label=f"{label}  (Δχ²@ε=1: {res['dchi2_at_1']:+.1f}, "
                  f"{res['significance']:.1f}σ)")
ax.axhline(0, color='k',    linestyle='--', linewidth=0.8, label='LCDM baseline')
ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8, label=r'1$\sigma$')
ax.axhline(4, color='gray', linestyle='-.', linewidth=0.8, label=r'2$\sigma$')
ax.axvline(1, color='black',linestyle='--', linewidth=1.0, alpha=0.5,
           label='Full Union3 DDE')
ax.set_xlabel(r'Template amplitude $\epsilon$', fontsize=12)
ax.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax.set_title('Profile likelihood comparison\nUnion3 DDE tilt vs BOSS DR14',
             fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_BASE, 'comparison_profile_likelihood.png'),
            dpi=150)
plt.close()
print("Saved comparison_profile_likelihood.png")

# ── Plot 2: Fisher S/N comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart: per-bin S/N for each configuration
ax = axes[0]
n_configs = len(CONFIGS)
for i, cfg in enumerate(CONFIGS):
    label = cfg['label']
    res   = all_fisher[label]
    zout  = res['zout']
    x     = np.arange(len(zout)) + i * 0.25
    ax.bar(x, res['sn_per_bin'],
           width=0.22, color=res['color'],
           alpha=0.8, label=label, edgecolor='k', linewidth=0.3)
ax.axhline(2, color='red',  linestyle='--', linewidth=1.2, label='2σ threshold')
ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8, label='1σ threshold')
ax.set_ylabel('S/N per redshift bin', fontsize=11)
ax.set_title('Per-bin Fisher S/N\nby configuration', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# Line chart: S/N vs data volume
ax = axes[1]
vols = np.logspace(0, 3, 200)
for label, res in all_fisher.items():
    ax.loglog(vols, res['SN'] * np.sqrt(vols),
              color=res['color'], linewidth=2,
              label=f"{label} (S/N={res['SN']:.3f})")
ax.axhline(2, color='red',  linestyle='--', linewidth=1.2,
           label='2σ threshold')
ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8)
for name, vf, col in [('BOSS DR14', 1, 'navy'),
                       ('DESI DR2',  2, 'slategray'),
                       ('DESI full', 10,'royalblue')]:
    ax.axvline(vf, color=col, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(vf*1.05, 0.08, name, fontsize=7, color=col, rotation=90)
ax.set_xlabel('Data volume relative to BOSS DR14', fontsize=11)
ax.set_ylabel('S/N', fontsize=11)
ax.set_title('Fisher S/N vs data volume\nby configuration', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, which='both')

plt.suptitle('Fisher forecast comparison — Union3 DDE tilt', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_BASE, 'comparison_fisher_forecast.png'),
            dpi=150)
plt.close()
print("Saved comparison_fisher_forecast.png")

# ── Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL SUMMARY TABLE")
print("="*70)
print(f"\n{'Config':25s}  {'min_z':>5s}  {'chi2_base':>10s}  "
      f"{'Δχ²(ε=1)':>10s}  {'sig(σ)':>8s}  "
      f"{'S/N BOSS':>9s}  {'S/N DR2':>8s}  {'Need Nx':>8s}")
print("-"*100)
for cfg in CONFIGS:
    label = cfg['label']
    pres  = all_profile[label]
    fres  = all_fisher[label]
    N     = fres['N_needed']
    print(f"{label:25s}  {cfg['min_z']:>5.1f}  "
          f"{pres['chi2_baseline']:>10.2f}  "
          f"{pres['dchi2_at_1']:>+10.2f}  "
          f"{pres['significance']:>8.2f}  "
          f"{fres['SN']:>9.4f}  "
          f"{fres['sn_desi_dr2']:>8.4f}  "
          f"{N:.1f}x" if N else f"{'<1x':>8s}")

# Save full summary
summary_file = os.path.join(RESULTS_BASE, 'full_summary.txt')
with open(summary_file, 'w') as f:
    f.write("Full analysis summary — Union3 DDE tilt vs BOSS DR14\n")
    f.write("="*70 + "\n\n")
    for cfg in CONFIGS:
        label = cfg['label']
        pres  = all_profile[label]
        fres  = all_fisher[label]
        f.write(f"Configuration: {label}\n")
        f.write(f"  Description:       {cfg['desc']}\n")
        f.write(f"  min_z:             {cfg['min_z']}\n")
        f.write(f"  Baseline chi2:     {pres['chi2_baseline']:.4f}\n")
        f.write(f"  Delta_chi2(eps=1): {pres['dchi2_at_1']:+.4f}\n")
        f.write(f"  Significance:      {pres['significance']:.3f} sigma\n")
        f.write(f"  p-value:           {pres['pval']:.6f}\n")
        f.write(f"  Preferred epsilon: {pres['best_eps']:.2f}\n")
        f.write(f"  S/N BOSS DR14:     {fres['SN']:.6f}\n")
        f.write(f"  S/N DESI DR2:      {fres['sn_desi_dr2']:.6f}\n")
        f.write(f"  S/N DESI full:     {fres['sn_desi_full']:.6f}\n")
        N = fres['N_needed']
        if N:
            f.write(f"  Data for 2-sigma:  {N:.1f}x BOSS DR14\n")
        f.write("\n")
print(f"\nFull summary saved to {summary_file}")
print("\nDONE")
