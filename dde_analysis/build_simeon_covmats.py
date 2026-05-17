"""
build_simeon_covmats.py

Build proposal covariance matrices from Simeon's converged chains.

Inputs:
  - Chain directory (default from load_chains.py):
      ~/InferenceLyaData/Chains/fps-meant

Outputs:
  - dde_analysis/results/covmats/simeon_lcdm_14p.covmat
  - dde_analysis/results/covmats/simeon_plus_epsilon_15p.covmat
  - dde_analysis/results/covmats/simeon_covmat_metadata.txt
"""

import glob
import os
import numpy as np


CHAIN_DIR = os.path.expanduser("~/InferenceLyaData/Chains/fps-meant")
OUTDIR = "/Users/helenagescu/lya_emulator/dde_analysis/results/covmats"
os.makedirs(OUTDIR, exist_ok=True)

OUT_LCDM = os.path.join(OUTDIR, "simeon_lcdm_14p.covmat")
OUT_EPS = os.path.join(OUTDIR, "simeon_plus_epsilon_15p.covmat")
OUT_META = os.path.join(OUTDIR, "simeon_covmat_metadata.txt")

# Chain columns used by existing load_chains.py:
# 0 weight, 1 minuslogpost, 2..15 params, ... 18 chi2
PARAM_COLS = list(range(2, 16))
PARAM_NAMES = [
    "dtau0",
    "tau0",
    "ns",
    "Ap",
    "herei",
    "heref",
    "alphaq",
    "hub",
    "omegamh2",
    "hireionz",
    "bhfeedback",
    "a_lls",
    "a_dla",
    "fSiIII",
]


def weighted_mean_cov(x, w):
    w = np.asarray(w, dtype=float)
    wsum = np.sum(w)
    if wsum <= 0:
        raise RuntimeError("Non-positive total weight.")
    w = w / wsum

    mu = np.average(x, axis=0, weights=w)
    xc = x - mu
    # Weighted covariance (MLE style for proposal covmat usage)
    cov = (xc * w[:, None]).T @ xc
    return mu, cov


def regularize_to_spd(cov, floor=1e-10):
    sym = 0.5 * (cov + cov.T)
    vals = np.linalg.eigvalsh(sym)
    min_eig = float(vals.min())
    if min_eig <= floor:
        shift = floor - min_eig
        sym = sym + np.eye(sym.shape[0]) * shift
    return sym


def save_cobaya_covmat(path, names, cov):
    with open(path, "w") as f:
        f.write("# " + " ".join(names) + "\n")
        np.savetxt(f, cov, fmt="%.12e")


def main():
    chain_files = sorted(glob.glob(os.path.join(CHAIN_DIR, "*.txt")))
    if not chain_files:
        chain_files = sorted(glob.glob(os.path.join(CHAIN_DIR, "*")))
    if not chain_files:
        raise FileNotFoundError(f"No chain files found in {CHAIN_DIR}")

    rows = []
    for cf in chain_files:
        try:
            arr = np.loadtxt(cf, comments="#")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            rows.append(arr)
        except Exception as exc:
            print(f"Skipping unreadable file: {cf} ({exc})")

    if not rows:
        raise RuntimeError("No readable chain arrays found.")

    chains = np.vstack(rows)
    weights = chains[:, 0]
    params = chains[:, PARAM_COLS]

    mu, cov14 = weighted_mean_cov(params, weights)
    cov14 = regularize_to_spd(cov14, floor=1e-10)
    save_cobaya_covmat(OUT_LCDM, PARAM_NAMES, cov14)

    # Extend to epsilon parameter as final dimension.
    # Start with diagonal prior-scale variance for epsilon proposal.
    sigma_eps = 0.20
    var_eps = sigma_eps ** 2
    cov15 = np.zeros((15, 15), dtype=float)
    cov15[:14, :14] = cov14
    cov15[14, 14] = var_eps
    cov15 = regularize_to_spd(cov15, floor=1e-10)
    save_cobaya_covmat(OUT_EPS, PARAM_NAMES + ["epsilon"], cov15)

    with open(OUT_META, "w") as f:
        f.write("Simeon chain covariance products\n")
        f.write("=" * 60 + "\n")
        f.write(f"CHAIN_DIR: {CHAIN_DIR}\n")
        f.write(f"n_files: {len(chain_files)}\n")
        f.write(f"n_samples_total: {chains.shape[0]}\n")
        f.write(f"n_columns_total: {chains.shape[1]}\n\n")
        f.write("LCDM 14-parameter order:\n")
        for i, n in enumerate(PARAM_NAMES):
            f.write(f"  [{i:2d}] {n}\n")
        f.write("\nOutput files:\n")
        f.write(f"  {OUT_LCDM}\n")
        f.write(f"  {OUT_EPS}\n")
        f.write("\nEpsilon extension:\n")
        f.write("  epsilon index = 14 (last)\n")
        f.write(f"  sigma_eps_init = {sigma_eps:.3f}\n")
        f.write(f"  var_eps_init   = {var_eps:.6f}\n\n")
        f.write("Weighted means (14p):\n")
        for n, v in zip(PARAM_NAMES, mu):
            f.write(f"  {n:12s} {v:.10e}\n")

    print(f"Saved: {OUT_LCDM}")
    print(f"Saved: {OUT_EPS}")
    print(f"Saved: {OUT_META}")


if __name__ == "__main__":
    main()

