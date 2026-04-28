"""
fisher_forecast.py

Fisher matrix forecast for detecting the Union3 DDE spectral tilt
from Garza et al. 2026 (arXiv:2601.00767) in Lyman-alpha forest
flux power spectrum data.

This script does NOT require best-fit parameters. It only uses:
  - The BOSS DR14 covariance matrix (from the emulator data files)
  - The DDE template T(k, z) digitised from Figure 9 of Garza et al.

Scientific question answered:
  Is the DDE spectral tilt detectable with BOSS DR14?
  If not, how much must sensitivity improve for a 2-sigma detection?
  Is DESI DR2 enough?

Usage:
    python dde_analysis/fisher_forecast.py
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

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results'
)
os.makedirs(OUTDIR, exist_ok=True)


# ── Step 1: Load likelihood and template ───────────────────────────────────
print("="*60)
print("Loading likelihood and template...")
print("="*60)

lik = LikelihoodClass(
    basedir=BASEDIR,
    mean_flux='s',
    min_z=2.2,
    max_z=4.6,
    optimise_GP=False,   # no need to train GP for Fisher forecast
    traindir=TRAINDIR,
    data_corr=True,
    sdss='dr14',
)

T_fn = make_template_interpolator()

print(f"Redshift bins: {lik.zout}")
print(f"k modes per bin: {len(lik.kf)}")
print(f"Total data points: {len(lik.zout) * len(lik.kf)}")


# ── Step 2: Build the template vector ─────────────────────────────────────
# T_vec is the DDE template evaluated at every (k, z) data point.
# Shape: (n_z * n_k,)
# T_vec[i] - 1 = fractional change in P(k,z) from DDE at point i.
# We use (T - 1) because we want the signal = change from LCDM,
# not the total prediction.
print("\nBuilding template vector...")

template_signal = []   # (T - 1) at each data point = the DDE signal
template_full   = []   # T at each data point = multiplicative factor

for bb, z in enumerate(lik.zout):
    # Evaluate template at the BOSS k-grid for this redshift bin
    T = T_fn(lik.kf, z, epsilon=1.0)
    template_signal.append(T - 1.0)   # fractional change
    template_full.append(T)

template_signal = np.concatenate(template_signal)  # shape (n_total,)
template_full   = np.concatenate(template_full)

print(f"Template vector shape: {template_signal.shape}")
print(f"Template range: [{template_signal.min():.4f}, "
      f"{template_signal.max():.4f}]")
print(f"  (positive = more power than LCDM at that scale/redshift)")
print(f"  (negative = less power than LCDM at that scale/redshift)")


# ── Step 3: Build the full block-diagonal covariance matrix ───────────────
# The BOSS DR14 covariance is stored per redshift bin.
# We assemble the full matrix as a block diagonal.
print("\nBuilding covariance matrix...")

cov_blocks = []
for bb in range(len(lik.zout)):
    C_bb = lik.get_BOSS_error(bb)
    cov_blocks.append(C_bb)

# Full block-diagonal covariance matrix
C_full = scipy.linalg.block_diag(*cov_blocks)
print(f"Full covariance matrix shape: {C_full.shape}")

# Compute inverse
C_inv = np.linalg.inv(C_full)
print("Covariance matrix inverted successfully.")

# Report diagonal (variance per data point) as a sanity check
variances = np.diag(C_full)
print(f"Diagonal variance range: [{variances.min():.3e}, "
      f"{variances.max():.3e}]")


# ── Step 4: Fisher signal-to-noise for BOSS DR14 ──────────────────────────
# (S/N)^2 = T^T C^{-1} T
# where T here is the signal vector (T_template - 1),
# i.e. the fractional change in P(k,z) from DDE.
#
# This is the OPTIMAL signal-to-noise -- the best you could achieve
# if you perfectly knew all other parameters and the only unknown
# was the DDE template amplitude epsilon.
print("\n" + "="*60)
print("Fisher signal-to-noise for BOSS DR14")
print("="*60)

SN_sq_boss = template_signal @ C_inv @ template_signal
SN_boss = np.sqrt(SN_sq_boss)

print(f"(S/N)^2  = {SN_sq_boss:.4f}")
print(f"S/N      = {SN_boss:.4f}")

if SN_boss < 1:
    print("Interpretation: BOSS DR14 is NOT sensitive to the DDE tilt.")
    print("  The signal is smaller than the noise. Cannot detect or")
    print("  rule out the DDE tilt with current data.")
elif SN_boss < 2:
    print("Interpretation: BOSS DR14 has MARGINAL sensitivity.")
    print("  The signal is present but below the 2-sigma threshold.")
else:
    print("Interpretation: BOSS DR14 COULD detect the DDE tilt.")
    print(f"  S/N = {SN_boss:.2f} > 2 sigma in principle.")


# ── Step 5: Per-redshift-bin contribution to S/N ──────────────────────────
# Which redshift bins drive the sensitivity?
print("\nPer-redshift-bin S/N contributions:")
n_k = len(lik.kf)
sn_per_bin = []
for bb, z in enumerate(lik.zout):
    T_bb  = template_signal[bb*n_k:(bb+1)*n_k]
    C_bb  = cov_blocks[bb]
    ic_bb = np.linalg.inv(C_bb)
    sn_sq_bb = T_bb @ ic_bb @ T_bb
    sn_per_bin.append(np.sqrt(sn_sq_bb))
    print(f"  z = {z:.3f}   S/N = {sn_per_bin[-1]:.4f}")

sn_per_bin = np.array(sn_per_bin)


# ── Step 6: How much improvement is needed for 2-sigma detection? ─────────
# If we scale the covariance by f (f < 1 = better data),
# then S/N scales as 1/sqrt(f), so:
#   new S/N = S/N_boss / sqrt(f)
# For new S/N = 2:
#   f = (S/N_boss / 2)^2
# Number of times more data needed = 1/f (since f = 1/N for N times more data)
print("\n" + "="*60)
print("Forecasting: how much improvement is needed?")
print("="*60)

f_needed = (SN_boss / 2.0)**2
N_times_more = 1.0 / f_needed if f_needed < 1 else None

print(f"Covariance scale factor f for 2-sigma detection: {f_needed:.4f}")
if N_times_more is not None:
    print(f"This requires {N_times_more:.1f}x more data than BOSS DR14")
    print(f"  (i.e. {N_times_more:.1f}x more quasar sightlines, "
          f"assuming noise-dominated)")
else:
    print("BOSS DR14 already exceeds 2-sigma sensitivity (f > 1 not needed)")


# ── Step 7: S/N as a function of data volume improvement ──────────────────
print("\nS/N as a function of data volume relative to BOSS DR14:")
volume_factors = np.array([1, 2, 5, 10, 20, 50, 100])
print(f"  {'Volume factor':>15s}   {'S/N':>8s}   {'Detectable at 2sig?':>20s}")
for vf in volume_factors:
    sn = SN_boss * np.sqrt(vf)
    detectable = "YES" if sn >= 2 else "no"
    print(f"  {vf:>15d}x   {sn:>8.3f}   {detectable:>20s}")

# DESI DR2 specifically
sn_desi_dr2 = SN_boss * np.sqrt(2)
print(f"\nDESI DR2 (2x BOSS):  S/N = {sn_desi_dr2:.4f}")
if sn_desi_dr2 >= 2:
    print("  -> DESI DR2 IS sufficient for a 2-sigma detection")
else:
    print("  -> DESI DR2 is NOT sufficient for a 2-sigma detection")
    print(f"     Would need {(2/sn_desi_dr2)**2:.1f}x more data than DESI DR2")


# ── Step 8: Plot 1 — S/N per redshift bin ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(range(len(lik.zout)), sn_per_bin,
       color='teal', alpha=0.8, edgecolor='k', linewidth=0.5)
ax.axhline(2, color='coral', linestyle='--',
           linewidth=1.5, label='2σ threshold')
ax.axhline(1, color='gray', linestyle=':',
           linewidth=1.0, label='1σ threshold')
ax.set_xticks(range(len(lik.zout)))
ax.set_xticklabels([f'z={z:.2f}' for z in lik.zout],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel('S/N contribution', fontsize=12)
ax.set_title('Per-redshift-bin S/N\n(Union3 DDE template, BOSS DR14)',
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# ── Step 9: Plot 2 — S/N vs data volume ───────────────────────────────────
ax = axes[1]
volume_range = np.logspace(0, 3, 100)
sn_range = SN_boss * np.sqrt(volume_range)

ax.loglog(volume_range, sn_range,
          color='teal', linewidth=2.5, label='S/N (Fisher forecast)')
ax.axhline(2, color='coral', linestyle='--',
           linewidth=1.5, label='2σ detection threshold')
ax.axhline(1, color='gray', linestyle=':',
           linewidth=1.0, label='1σ threshold')

# Mark specific surveys
surveys = {
    'BOSS DR14': 1,
    'DESI DR2':  2,
    'DESI full': 10,
}
colors_survey = ['navy', 'steelblue', 'royalblue']
for (name, vf), col in zip(surveys.items(), colors_survey):
    sn_s = SN_boss * np.sqrt(vf)
    ax.axvline(vf, color=col, linestyle=':', linewidth=1.0, alpha=0.7)
    ax.plot(vf, sn_s, 'o', color=col, markersize=8, label=f'{name} (S/N={sn_s:.2f})')

ax.set_xlabel('Data volume relative to BOSS DR14', fontsize=12)
ax.set_ylabel('S/N', fontsize=12)
ax.set_title('Fisher forecast: S/N vs data volume\n(Union3 DDE tilt)',
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, which='both')

plt.tight_layout()
plotfile = os.path.join(OUTDIR, 'fisher_forecast_union3.png')
plt.savefig(plotfile, dpi=150)
print(f"\nPlot saved to {plotfile}")
plt.show()


# ── Step 10: Save summary ──────────────────────────────────────────────────
summary = {
    'SN_boss_dr14': SN_boss,
    'SN_desi_dr2':  sn_desi_dr2,
    'f_needed_for_2sigma': f_needed,
    'N_times_more_data_needed': N_times_more if N_times_more else 'already detectable',
}

outfile = os.path.join(OUTDIR, 'fisher_forecast_summary.txt')
with open(outfile, 'w') as f:
    f.write("Fisher forecast summary — Union3 DDE tilt\n")
    f.write("="*50 + "\n")
    f.write(f"S/N with BOSS DR14:          {SN_boss:.4f}\n")
    f.write(f"S/N with DESI DR2 (2x):      {sn_desi_dr2:.4f}\n")
    f.write(f"Covariance scale for 2sigma: {f_needed:.4f}\n")
    if N_times_more:
        f.write(f"Data volume for 2sigma:      {N_times_more:.1f}x BOSS DR14\n")
    f.write("\nPer-redshift-bin S/N:\n")
    for z, sn in zip(lik.zout, sn_per_bin):
        f.write(f"  z = {z:.3f}   S/N = {sn:.4f}\n")

print(f"Summary saved to {outfile}")
print("\n" + "="*60)
print("DONE")
print("="*60)
