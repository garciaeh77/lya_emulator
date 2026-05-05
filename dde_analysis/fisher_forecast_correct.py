"""
fisher_forecast_correct.py

CORRECTED Fisher forecast with proper unit handling.

The Garza et al. template is dimensionless:
    ratio(k,z) = [Delta^2_DDE / Delta^2_LCDM] - 1

The BOSS DR14 covariance C has units (km/s)^2
(covariance of P(k) measurements in km/s units)

The correctly dimensioned signal vector is:
    T(k,z) [km/s] = P_bestfit(k,z) [km/s] * ratio(k,z) [dimensionless]

So we DO need the best-fit P(k) to do the forecast correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats
import os, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.template import make_template_interpolator

BASEDIR  = '/Users/helenagescu/lya_emulator/dtau-48-48'
TRAINDIR = '/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf'
OUTDIR   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'fisher_corrected'
)
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    {'label': 'all_z',           'min_z': 2.2,
     'params_file': 'bestfit_z2.2-4.6.txt', 'color': 'teal'},
    {'label': 'no_z2p2',         'min_z': 2.4,
     'params_file': 'bestfit_z2.6-4.6.txt', 'color': 'steelblue'},
    {'label': 'no_z2p2_no_z2p4', 'min_z': 2.6,
     'params_file': 'bestfit_z2.6-4.6.txt', 'color': 'coral'},
]

T_fn = make_template_interpolator()

print("="*65)
print("CORRECTED Fisher forecast — with proper unit handling")
print("="*65)
print()
print("Signal vector: T(k,z) = P_bestfit(k,z) * (ratio(k,z))")
print("Units: T in km/s, C in (km/s)^2, T^T C^-1 T dimensionless")
print()

all_results = {}


def _split_parameter_vectors(full_params):
    """
    Split full likelihood parameters into:
      - gp_params: parameters accepted by LikelihoodClass.get_predicted
      - full_params: full vector including nuisance corrections

    Current files store 14 parameters:
      dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback a_lls a_dla fSiIII
    get_predicted expects only the first 11 (up to bhfeedback).
    """
    p = np.asarray(full_params).reshape(-1)
    if p.size < 11:
        raise ValueError(
            f"Expected at least 11 parameters, got {p.size}: {p}"
        )
    return p[:11], p


for cfg in CONFIGS:
    label      = cfg['label']
    min_z      = cfg['min_z']
    pfile      = os.path.join('dde_analysis/results', cfg['params_file'])

    print(f"{'='*50}")
    print(f"Configuration: {label}  (min_z={min_z})")
    print(f"{'='*50}")

    # Load likelihood
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lik = LikelihoodClass(
            basedir=BASEDIR, mean_flux='s',
            min_z=min_z, max_z=4.6,
            optimise_GP=True, traindir=TRAINDIR,
            data_corr=True, sdss='dr14',
        )

    # Load best-fit parameters
    params = np.loadtxt(pfile)
    params_gp, params_full = _split_parameter_vectors(params)
    print(f"Loaded params from {cfg['params_file']}")
    print(f"  Using {params_gp.size} params for emulator prediction")
    if params_full.size > params_gp.size:
        print(f"  Found {params_full.size - params_gp.size} nuisance params for data corrections")

    # Get best-fit P(k) prediction in km/s units
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        okf, predicted, std = lik.get_predicted(params_gp)

    # Apply nuisance data corrections if available in the full parameter vector.
    # This makes P_bestfit consistent with the model compared against BOSS data.
    if lik.data_corr and params_full.size >= 14:
        for bb, z in enumerate(lik.zout):
            corr = lik.get_data_correction(okf[bb], params_full, z)
            predicted[bb] = predicted[bb] * corr

    print(f"Redshift bins: {lik.zout}")
    print(f"k modes per bin: {len(okf[0])}")
    print(f"P(k) range: [{np.min([p.min() for p in predicted]):.4f},"
          f" {np.max([p.max() for p in predicted]):.4f}] km/s")

    # Build correctly dimensioned signal vector
    # T(k,z) [km/s] = P_bestfit(k,z) [km/s] * (ratio(k,z) [dimensionless])
    signal_vec = []
    ratio_vec  = []
    for bb, z in enumerate(lik.zout):
        ratio   = T_fn(okf[bb], z, epsilon=1.0) - 1.0  # dimensionless
        T_phys  = predicted[bb] * ratio                  # km/s
        signal_vec.append(T_phys)
        ratio_vec.append(ratio)

    signal = np.concatenate(signal_vec)   # km/s
    ratio  = np.concatenate(ratio_vec)    # dimensionless

    print(f"\nDimensionless ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")
    print(f"Physical signal T range:   [{signal.min():.6f},"
          f" {signal.max():.6f}] km/s")

    # Build block-diagonal covariance matrix (km/s)^2
    n_k        = len(okf[0])
    cov_blocks = []
    for bb in range(len(lik.zout)):
        C_bb = lik.get_BOSS_error(bb)
        # Match k grid
        idp = np.where(lik.kf >= okf[bb][0])[0]
        cov_blocks.append(C_bb[np.ix_(idp, idp)])

    C_full = scipy.linalg.block_diag(*cov_blocks)
    C_inv  = np.linalg.inv(C_full)

    print(f"Covariance diagonal range: [{np.diag(C_full).min():.4e},"
          f" {np.diag(C_full).max():.4e}] (km/s)^2")
    print(f"Signal^2 / variance range: "
          f"[{(signal**2 / np.diag(C_full)).min():.6e},"
          f" {(signal**2 / np.diag(C_full)).max():.6e}]  (dimensionless)")

    # Fisher S/N — now dimensionally consistent
    SN_sq = float(signal @ C_inv @ signal)
    SN    = np.sqrt(SN_sq)

    # Per-bin S/N
    sn_per_bin = []
    offset = 0
    for bb, z in enumerate(lik.zout):
        n  = len(signal_vec[bb])
        s  = signal[offset:offset+n]
        Ci = C_inv[offset:offset+n, offset:offset+n]
        sn_per_bin.append(np.sqrt(float(s @ Ci @ s)))
        offset += n
    sn_per_bin = np.array(sn_per_bin)

    # Forecasts
    sn_desi_dr2  = SN * np.sqrt(2)
    sn_desi_full = SN * np.sqrt(10)
    N_needed     = (2.0 / SN)**2 if SN > 0 else np.inf

    print(f"\nCORRECTED Fisher results:")
    print(f"  (S/N)^2       = {SN_sq:.6f}")
    print(f"  S/N           = {SN:.6f}")
    print(f"  S/N DESI DR2  = {sn_desi_dr2:.6f}")
    print(f"  S/N DESI full = {sn_desi_full:.6f}")
    print(f"  Need Nx       = {N_needed:.1f}x BOSS DR14")
    print(f"\n  Per-bin S/N:")
    for z, sn in zip(lik.zout, sn_per_bin):
        print(f"    z={z:.2f}  S/N={sn:.6f}")

    all_results[label] = {
        'SN': SN, 'sn_desi_dr2': sn_desi_dr2,
        'sn_desi_full': sn_desi_full, 'N_needed': N_needed,
        'sn_per_bin': sn_per_bin, 'zout': lik.zout.copy(),
        'color': cfg['color'],
    }

    # Save per-config summary
    outfile = os.path.join(OUTDIR, f'fisher_corrected_{label}.txt')
    with open(outfile, 'w') as f:
        f.write(f"Corrected Fisher forecast — {label}\n{'='*50}\n")
        f.write(f"min_z = {min_z}\n")
        f.write(f"Signal: T(k,z) = P_bestfit(k,z) * ratio(k,z)\n\n")
        f.write(f"S/N BOSS DR14:       {SN:.6f}\n")
        f.write(f"S/N DESI DR2 (2x):   {sn_desi_dr2:.6f}\n")
        f.write(f"S/N DESI full (10x): {sn_desi_full:.6f}\n")
        f.write(f"Need Nx:             {N_needed:.1f}x BOSS DR14\n\n")
        f.write("Per-bin S/N:\n")
        for z, sn in zip(lik.zout, sn_per_bin):
            f.write(f"  z={z:.3f}  S/N={sn:.6f}\n")
    print(f"\nSaved to {outfile}")
    print()

# ── Summary comparison ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY — CORRECTED vs PREVIOUS (incorrect) Fisher results")
print("="*65)

previous_SN = {'all_z': 0.2287, 'no_z2p2': 0.2024,
               'no_z2p2_no_z2p4': 0.1742}

print(f"\n{'Config':25s}  {'S/N (wrong)':>12s}  {'S/N (correct)':>14s}"
      f"  {'Ratio':>8s}  {'Need Nx':>8s}")
print("-"*75)
for label, res in all_results.items():
    old = previous_SN.get(label, 0)
    new = res['SN']
    ratio = new/old if old > 0 else float('inf')
    print(f"{label:25s}  {old:>12.4f}  {new:>14.6f}"
          f"  {ratio:>8.3f}  {res['N_needed']:>8.1f}x")

# ── Comparison plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# S/N vs data volume
ax = axes[0]
vols = np.logspace(0, 4, 200)
for label, res in all_results.items():
    ax.loglog(vols, res['SN'] * np.sqrt(vols),
              color=res['color'], linewidth=2, label=label)
# Previous (wrong) results as dashed
for label, sn_old in previous_SN.items():
    col = all_results[label]['color']
    ax.loglog(vols, sn_old * np.sqrt(vols),
              color=col, linewidth=1.5, linestyle='--',
              alpha=0.5, label=f'{label} (prev)')

ax.axhline(2, color='red',  linestyle='--', linewidth=1.5, label='2σ')
ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8)
for name, vf, col in [('BOSS DR14', 1, 'navy'),
                       ('DESI DR2',  2, 'steelblue'),
                       ('DESI full', 10,'royalblue')]:
    ax.axvline(vf, color=col, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(vf*1.1, 0.001, name, fontsize=7, color=col, rotation=90)
ax.set_xlabel('Data volume relative to BOSS DR14', fontsize=11)
ax.set_ylabel('S/N', fontsize=11)
ax.set_title('Corrected Fisher forecast\n(solid=correct, dashed=previous)', fontsize=10)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, which='both')

# Per-bin S/N bar chart
ax = axes[1]
for i, (label, res) in enumerate(all_results.items()):
    zout = res['zout']
    x    = np.arange(len(zout)) + i * 0.28
    ax.bar(x, res['sn_per_bin'], width=0.25,
           color=res['color'], alpha=0.8,
           edgecolor='k', linewidth=0.3, label=label)
ax.axhline(2, color='red',  linestyle='--', linewidth=1.2)
ax.axhline(1, color='gray', linestyle=':',  linewidth=0.8)
# Use first config zout for x labels
first_zout = list(all_results.values())[0]['zout']
ax.set_xticks(np.arange(len(first_zout)) + 0.28)
ax.set_xticklabels([f'z={z:.1f}' for z in first_zout],
                   rotation=45, ha='right', fontsize=7)
ax.set_ylabel('S/N per redshift bin', fontsize=11)
ax.set_title('Corrected per-bin S/N\nby configuration', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('Corrected Fisher forecast — Union3 DDE tilt vs BOSS DR14',
             fontsize=12)
plt.tight_layout()
plotfile = os.path.join(OUTDIR, 'fisher_corrected_comparison.png')
plt.savefig(plotfile, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {plotfile}")
print("\nDONE")
