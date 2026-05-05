"""
fisher_forecast_data_vector.py

Full P1D Fisher forecast for the Union3 DDE template using the
observed BOSS/eBOSS data vector as the fiducial P(k, z), instead of
an emulator best-fit model.

Signal construction:
    ratio(k, z) = T_union3(k, z, epsilon=1) - 1      [dimensionless]
    deltaP(k, z) = P_data(k, z) * ratio(k, z)        [km/s]

Fisher:
    (S/N)^2 = deltaP^T C^{-1} deltaP

Notes:
  - This is dimensionally consistent (deltaP in km/s, C in (km/s)^2).
  - This is a fixed-template-amplitude forecast (no parameter profiling).
  - Using P_data as fiducial is convenient but noisier than using a smooth
    best-fit model.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.template import make_template_interpolator

BASEDIR = "/Users/helenagescu/lya_emulator/dtau-48-48"
TRAINDIR = "/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf"
OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "fisher_data_vector"
)
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    {"label": "all_z", "min_z": 2.2, "color": "teal"},
    {"label": "no_z2p2", "min_z": 2.4, "color": "steelblue"},
    {"label": "no_z2p2_no_z2p4", "min_z": 2.6, "color": "coral"},
]

T_fn = make_template_interpolator()

print("=" * 68)
print("Full P1D Fisher using BOSS/eBOSS data vector as fiducial P(k,z)")
print("=" * 68)
print("Signal: deltaP = P_data * (T - 1)")
print("Units:  deltaP [km/s], C [(km/s)^2], deltaP^T C^-1 deltaP [dimensionless]")
print()

all_results = {}

for cfg in CONFIGS:
    label = cfg["label"]
    min_z = cfg["min_z"]

    print(f"{'=' * 52}")
    print(f"Configuration: {label}  (min_z={min_z})")
    print(f"{'=' * 52}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lik = LikelihoodClass(
            basedir=BASEDIR,
            mean_flux="s",
            min_z=min_z,
            max_z=4.6,
            optimise_GP=False,
            traindir=TRAINDIR,
            data_corr=True,
            sdss="dr14",
        )

    # Observed BOSS/eBOSS data vector in km/s, shape (n_z, n_k)
    p_data = np.array(lik.BOSS_flux_power, copy=True)
    n_nonpos = int(np.count_nonzero(p_data <= 0))
    if n_nonpos > 0:
        print(f"WARNING: found {n_nonpos} non-positive P_data points.")

    # Build dimensionless ratio and dimensioned deltaP
    ratio_vec = []
    signal_vec = []
    for bb, z in enumerate(lik.zout):
        ratio = T_fn(lik.kf, z, epsilon=1.0) - 1.0
        delta_p = p_data[bb] * ratio
        ratio_vec.append(ratio)
        signal_vec.append(delta_p)

    ratio_flat = np.concatenate(ratio_vec)
    signal = np.concatenate(signal_vec)

    print(f"Redshift bins: {lik.zout}")
    print(f"k modes per bin: {len(lik.kf)}")
    print(f"P_data range: [{p_data.min():.4f}, {p_data.max():.4f}] km/s")
    print(
        f"Template ratio range: [{ratio_flat.min():.4f}, {ratio_flat.max():.4f}]"
    )
    print(f"deltaP range: [{signal.min():.6f}, {signal.max():.6f}] km/s")

    cov_blocks = [lik.get_BOSS_error(bb) for bb in range(len(lik.zout))]
    c_full = scipy.linalg.block_diag(*cov_blocks)
    c_inv = np.linalg.inv(c_full)

    print(
        f"Covariance diagonal range: "
        f"[{np.diag(c_full).min():.4e}, {np.diag(c_full).max():.4e}] (km/s)^2"
    )

    sn_sq = float(signal @ c_inv @ signal)
    sn = np.sqrt(sn_sq)
    n_needed = (2.0 / sn) ** 2 if sn > 0 else np.inf

    # Per-bin S/N (using each z-block independently)
    sn_per_bin = []
    n_k = len(lik.kf)
    for bb in range(len(lik.zout)):
        s_bb = signal[bb * n_k : (bb + 1) * n_k]
        ci_bb = np.linalg.inv(cov_blocks[bb])
        sn_per_bin.append(np.sqrt(float(s_bb @ ci_bb @ s_bb)))
    sn_per_bin = np.array(sn_per_bin)

    print("\nData-vector Fisher results:")
    print(f"  (S/N)^2       = {sn_sq:.6f}")
    print(f"  S/N           = {sn:.6f}")
    print(f"  S/N DESI DR2  = {sn * np.sqrt(2):.6f}")
    print(f"  S/N DESI full = {sn * np.sqrt(10):.6f}")
    print(f"  Need Nx       = {n_needed:.1f}x BOSS DR14")
    print("  Per-bin S/N:")
    for z, sn_z in zip(lik.zout, sn_per_bin):
        print(f"    z={z:.2f}  S/N={sn_z:.6f}")
    print()

    all_results[label] = {
        "SN": sn,
        "N_needed": n_needed,
        "sn_per_bin": sn_per_bin,
        "zout": np.array(lik.zout),
        "color": cfg["color"],
    }

    txt_file = os.path.join(OUTDIR, f"fisher_data_vector_{label}.txt")
    with open(txt_file, "w") as f:
        f.write(f"Data-vector Fisher forecast - {label}\n")
        f.write("=" * 54 + "\n")
        f.write(f"min_z = {min_z}\n")
        f.write("Signal = P_data * (T - 1)\n\n")
        f.write(f"S/N BOSS DR14:       {sn:.6f}\n")
        f.write(f"S/N DESI DR2 (2x):   {sn * np.sqrt(2):.6f}\n")
        f.write(f"S/N DESI full (10x): {sn * np.sqrt(10):.6f}\n")
        f.write(f"Need Nx:             {n_needed:.1f}x BOSS DR14\n\n")
        f.write("Per-bin S/N:\n")
        for z, sn_z in zip(lik.zout, sn_per_bin):
            f.write(f"  z={z:.3f}  S/N={sn_z:.6f}\n")

# Plot S/N vs data volume
fig, ax = plt.subplots(figsize=(7, 5))
vols = np.logspace(0, 4, 200)
for label, res in all_results.items():
    ax.loglog(
        vols,
        res["SN"] * np.sqrt(vols),
        color=res["color"],
        linewidth=2,
        label=f"{label} (S/N={res['SN']:.3f})",
    )
ax.axhline(2, color="red", linestyle="--", linewidth=1.2, label="2 sigma")
ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Data volume relative to BOSS DR14")
ax.set_ylabel("S/N")
ax.set_title("Full P1D Fisher with data-vector fiducial")
ax.grid(True, alpha=0.2, which="both")
ax.legend(fontsize=8)
plt.tight_layout()

plot_file = os.path.join(OUTDIR, "fisher_data_vector_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"Saved plot: {plot_file}")

print("\nSummary table")
print("-" * 68)
print(f"{'Config':25s}  {'S/N':>10s}  {'Need Nx':>10s}")
for label, res in all_results.items():
    print(f"{label:25s}  {res['SN']:>10.4f}  {res['N_needed']:>9.1f}x")
print("-" * 68)
print("DONE")
