"""
fisher_forecast_allz.py

Fisher matrix forecast for detecting the Union3 DDE spectral tilt
using ALL redshift bins (z = 2.2 to 4.6).

Results saved to dde_analysis/results/all_zbins/
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.template import make_template_interpolator

# ── Configuration ──────────────────────────────────────────────────────────
BASEDIR  = '/Users/helenagescu/lya_emulator/dtau-48-48'
TRAINDIR = '/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf'
MIN_Z    = 2.2
MAX_Z    = 4.6
LABEL    = 'all_zbins'
OUTDIR   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', LABEL
)
os.makedirs(OUTDIR, exist_ok=True)

# ── Load likelihood and template ───────────────────────────────────────────
print("="*60)
print(f"Fisher forecast — {LABEL} (min_z={MIN_Z})")
print("="*60)

lik = LikelihoodClass(
    basedir=BASEDIR, mean_flux='s',
    min_z=MIN_Z, max_z=MAX_Z,
    optimise_GP=False, traindir=TRAINDIR,
    data_corr=True, sdss='dr14',
)

T_fn = make_template_interpolator()

print(f"Redshift bins: {lik.zout}")
print(f"k modes per bin: {len(lik.kf)}")
print(f"Total data points: {len(lik.zout) * len(lik.kf)}")

# ── Build template vector ──────────────────────────────────────────────────
print("\nBuilding template vector...")
template_signal = []
for bb, z in enumerate(lik.zout):
    T = T_fn(lik.kf, z, epsilon=1.0)
    template_signal.append(T - 1.0)
template_signal = np.concatenate(template_signal)

print(f"Template vector shape: {template_signal.shape}")
print(f"Template range: [{template_signal.min():.4f}, {template_signal.max():.4f}]")

# ── Build covariance matrix ────────────────────────────────────────────────
print("\nBuilding covariance matrix...")
cov_blocks = [lik.get_BOSS_error(bb) for bb in range(len(lik.zout))]
C_full = scipy.linalg.block_diag(*cov_blocks)
C_inv  = np.linalg.inv(C_full)
print(f"Covariance matrix shape: {C_full.shape}")

# ── Fisher S/N ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Fisher signal-to-noise")
print("="*60)

SN_sq = template_signal @ C_inv @ template_signal
SN    = np.sqrt(SN_sq)

print(f"(S/N)^2 = {SN_sq:.6f}")
print(f"S/N     = {SN:.6f}")

if SN < 1:
    print("Interpretation: NOT sensitive to the DDE tilt with current data.")
elif SN < 2:
    print("Interpretation: MARGINAL sensitivity — below 2-sigma threshold.")
else:
    print(f"Interpretation: DETECTABLE at {SN:.1f} sigma.")

# ── Per-bin S/N ───────────────────────────────────────────────────────────
print("\nPer-redshift-bin S/N contributions:")
n_k = len(lik.kf)
sn_per_bin = []
for bb, z in enumerate(lik.zout):
    T_bb  = template_signal[bb*n_k:(bb+1)*n_k]
    ic_bb = np.linalg.inv(cov_blocks[bb])
    sn_bb = np.sqrt(T_bb @ ic_bb @ T_bb)
    sn_per_bin.append(sn_bb)
    print(f"  z = {z:.3f}   S/N = {sn_bb:.6f}")
sn_per_bin = np.array(sn_per_bin)

# ── Forecasting ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Data volume needed for 2-sigma detection")
print("="*60)

f_needed     = (SN / 2.0)**2
N_needed     = 1.0 / f_needed if f_needed < 1 else None
sn_desi_dr2  = SN * np.sqrt(2)
sn_desi_full = SN * np.sqrt(10)

print(f"Covariance scale factor for 2-sigma: {f_needed:.6f}")
if N_needed:
    print(f"Data volume needed: {N_needed:.1f}x BOSS DR14")
print(f"\nS/N forecasts:")
print(f"  BOSS DR14:          {SN:.4f}")
print(f"  DESI DR2  (2x):     {sn_desi_dr2:.4f}")
print(f"  DESI full (10x):    {sn_desi_full:.4f}")
if N_needed:
    print(f"  Need {N_needed:.0f}x BOSS for 2-sigma")

# ── Volume scan table ─────────────────────────────────────────────────────
print(f"\n{'Volume':>10s}  {'S/N':>8s}  {'Detectable?':>12s}")
for vf in [1, 2, 5, 10, 20, 50, 100, 200]:
    sn_v = SN * np.sqrt(vf)
    det  = "YES" if sn_v >= 2 else "no"
    print(f"  {vf:6d}x    {sn_v:8.4f}  {det:>12s}")

# ── Plot 1: per-bin bar chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
colors = ['teal' if z > 2.2 else 'coral' for z in lik.zout]
bars = ax.bar(range(len(lik.zout)), sn_per_bin,
              color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
ax.axhline(2, color='red',  linestyle='--', linewidth=1.5, label='2σ threshold')
ax.axhline(1, color='gray', linestyle=':',  linewidth=1.0, label='1σ threshold')
ax.set_xticks(range(len(lik.zout)))
ax.set_xticklabels([f'z={z:.2f}' for z in lik.zout],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel('S/N contribution', fontsize=12)
ax.set_title(f'Per-redshift-bin S/N\n({LABEL}, total S/N={SN:.4f})', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# ── Plot 2: S/N vs data volume ────────────────────────────────────────────
ax = axes[1]
vols = np.logspace(0, 3, 200)
ax.loglog(vols, SN * np.sqrt(vols),
          color='teal', linewidth=2.5, label=f'S/N ({LABEL})')
ax.axhline(2, color='red',  linestyle='--', linewidth=1.5,
           label='2σ threshold')
ax.axhline(1, color='gray', linestyle=':',  linewidth=1.0,
           label='1σ threshold')

for name, vf, col in [('BOSS DR14', 1, 'navy'),
                       ('DESI DR2',  2, 'steelblue'),
                       ('DESI full', 10, 'royalblue')]:
    sn_s = SN * np.sqrt(vf)
    ax.plot(vf, sn_s, 'o', color=col, markersize=8,
            label=f'{name} (S/N={sn_s:.3f})')

ax.set_xlabel('Data volume relative to BOSS DR14', fontsize=12)
ax.set_ylabel('S/N', fontsize=12)
ax.set_title('Fisher forecast: S/N vs data volume\n(Union3 DDE tilt)',
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, which='both')

plt.suptitle(f'Fisher forecast — {LABEL}', fontsize=13, y=1.01)
plt.tight_layout()
plotfile = os.path.join(OUTDIR, 'fisher_forecast.png')
plt.savefig(plotfile, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {plotfile}")

# ── Save summary ──────────────────────────────────────────────────────────
outfile = os.path.join(OUTDIR, 'fisher_summary.txt')
with open(outfile, 'w') as f:
    f.write(f"Fisher forecast summary — Union3 DDE tilt ({LABEL})\n")
    f.write("="*50 + "\n")
    f.write(f"min_z = {MIN_Z}\n")
    f.write(f"max_z = {MAX_Z}\n")
    f.write(f"Total data points: {len(lik.zout)*len(lik.kf)}\n\n")
    f.write(f"S/N with BOSS DR14:          {SN:.6f}\n")
    f.write(f"S/N with DESI DR2 (2x):      {sn_desi_dr2:.6f}\n")
    f.write(f"S/N with DESI full (10x):    {sn_desi_full:.6f}\n")
    if N_needed:
        f.write(f"Data volume for 2-sigma:     {N_needed:.1f}x BOSS DR14\n")
    f.write("\nPer-redshift-bin S/N:\n")
    for z, sn in zip(lik.zout, sn_per_bin):
        f.write(f"  z = {z:.3f}   S/N = {sn:.6f}\n")

print(f"Summary saved to {outfile}")
print("\n" + "="*60)
print("DONE")
print("="*60)
