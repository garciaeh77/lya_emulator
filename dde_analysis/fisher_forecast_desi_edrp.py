"""
fisher_forecast_desi_edrp.py

Full P1D Fisher forecast using DESI EDRP QMLE data directly
(instead of BOSS/eBOSS), with unit-consistent signal:

    deltaP_i = P_data_i * (T_i - 1)
    (S/N)^2  = deltaP^T C^{-1} deltaP

Data source:
  lyaemu/data/desi_edrp_qmle_data/
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dde_analysis.template import make_template_interpolator

DATA_FILE = (
    "/Users/helenagescu/lya_emulator/lyaemu/data/desi_edrp_qmle_data/"
    "desi-edrp-lyasb1subt-p1d-detailed-results.txt"
)
COV_FILE = (
    "/Users/helenagescu/lya_emulator/lyaemu/data/desi_edrp_qmle_data/"
    "desi-edrp-lyasb1subt-cov-total-offdiag-results.txt"
)

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "fisher_desi_edrp"
)
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    {"label": "all_z", "min_z": 2.0, "drop_z": [], "color": "teal"},
    {"label": "no_z2p2", "min_z": 2.0, "drop_z": [2.2], "color": "steelblue"},
    {
        "label": "no_z2p2_no_z2p4",
        "min_z": 2.0,
        "drop_z": [2.2, 2.4],
        "color": "coral",
    },
]

T_fn = make_template_interpolator()


def _mask_for_config(zvals, min_z, drop_z):
    mask = zvals >= min_z
    for z0 in drop_z:
        mask &= ~np.isclose(zvals, z0, atol=1e-6)
    return mask


print("=" * 70)
print("Full P1D Fisher with DESI EDRP QMLE data")
print("=" * 70)
print("Signal: deltaP = P_data * (T - 1)")
print("Units: deltaP [km/s], C [(km/s)^2]")
print()

arr = np.loadtxt(DATA_FILE)
cov = np.loadtxt(COV_FILE)

z = arr[:, 0]
k = arr[:, 3]  # kc
p_data = arr[:, 6]  # p_final

print(f"Loaded DESI rows: {arr.shape[0]}")
print(f"Covariance shape: {cov.shape}")
print(f"Unique z bins: {np.unique(z)}")
print(f"P_data range: [{p_data.min():.4f}, {p_data.max():.4f}] km/s")
print()

all_results = {}

for cfg in CONFIGS:
    label = cfg["label"]
    mask = _mask_for_config(z, cfg["min_z"], cfg["drop_z"])
    idx = np.where(mask)[0]

    z_sel = z[idx]
    k_sel = k[idx]
    p_sel = p_data[idx]
    c_sel = cov[np.ix_(idx, idx)]

    ratio = T_fn(k_sel, z_sel, epsilon=1.0) - 1.0
    delta_p = p_sel * ratio

    c_inv = np.linalg.inv(c_sel)
    sn_sq = float(delta_p @ c_inv @ delta_p)
    sn = np.sqrt(sn_sq)
    n_needed = (2.0 / sn) ** 2 if sn > 0 else np.inf

    # Per-z S/N from each redshift sub-block
    sn_per_z = []
    z_unique = np.unique(z_sel)
    for zz in z_unique:
        iz = np.where(np.isclose(z_sel, zz, atol=1e-6))[0]
        s_zz = delta_p[iz]
        c_zz = c_sel[np.ix_(iz, iz)]
        sn_zz = np.sqrt(float(s_zz @ np.linalg.inv(c_zz) @ s_zz))
        sn_per_z.append((zz, sn_zz))

    print(f"Config: {label}")
    print(f"  n_data points: {len(idx)}")
    print(f"  z bins: {z_unique}")
    print(f"  ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")
    print(f"  deltaP range: [{delta_p.min():.6f}, {delta_p.max():.6f}] km/s")
    print(f"  S/N: {sn:.6f}")
    print(f"  Need Nx: {n_needed:.2f}x DESI EDRP")
    print()

    all_results[label] = {
        "SN": sn,
        "N_needed": n_needed,
        "color": cfg["color"],
        "sn_per_z": sn_per_z,
    }

    out_txt = os.path.join(OUTDIR, f"fisher_desi_edrp_{label}.txt")
    with open(out_txt, "w") as f:
        f.write(f"DESI EDRP Fisher forecast - {label}\n")
        f.write("=" * 56 + "\n")
        f.write("Signal: deltaP = P_data * (T-1)\n\n")
        f.write(f"(S/N)^2 = {sn_sq:.6f}\n")
        f.write(f"S/N     = {sn:.6f}\n")
        f.write(f"Need Nx = {n_needed:.2f}x DESI EDRP\n\n")
        f.write("Per-z S/N:\n")
        for zz, sn_zz in sn_per_z:
            f.write(f"  z={zz:.2f}  S/N={sn_zz:.6f}\n")

# Comparison plot
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
ax.set_xlabel("Data volume relative to DESI EDRP")
ax.set_ylabel("S/N")
ax.set_title("DESI EDRP Fisher forecast (full P1D)")
ax.grid(True, alpha=0.2, which="both")
ax.legend(fontsize=8)
plt.tight_layout()
plot_path = os.path.join(OUTDIR, "fisher_desi_edrp_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")

print("Summary table")
print("-" * 68)
print(f"{'Config':25s}  {'S/N':>10s}  {'Need Nx':>10s}")
for label, res in all_results.items():
    print(f"{label:25s}  {res['SN']:>10.4f}  {res['N_needed']:>9.2f}x")
print("-" * 68)
print(f"Saved outputs in: {OUTDIR}")
print("DONE")
