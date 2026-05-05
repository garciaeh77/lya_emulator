"""
fisher_forecast_bestfit_chains.py

Full P1D Fisher forecast using the best-fit parameter vector extracted
from chains (bestfit_from_chains.txt), with dimensionally consistent
signal construction:

    deltaP(k,z) = P_bestfit(k,z) * (T(k,z)-1)
    (S/N)^2     = deltaP^T C^{-1} deltaP
"""

import os
import sys
import warnings

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass
from dde_analysis.template import make_template_interpolator

BASEDIR = "/Users/helenagescu/lya_emulator/dtau-48-48"
TRAINDIR = "/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf"
OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "fisher_chains_bestfit"
)
os.makedirs(OUTDIR, exist_ok=True)

# Chain best-fit currently corresponds to all-z setup
MIN_Z = 2.2
MAX_Z = 4.6
PARAMS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "bestfit_from_chains.txt"
)

T_fn = make_template_interpolator()


def _split_parameter_vectors(full_params):
    p = np.asarray(full_params).reshape(-1)
    if p.size < 11:
        raise ValueError(f"Expected at least 11 parameters, got {p.size}")
    return p[:11], p


print("=" * 68)
print("Full P1D Fisher forecast using bestfit_from_chains")
print("=" * 68)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lik = LikelihoodClass(
        basedir=BASEDIR,
        mean_flux="s",
        min_z=MIN_Z,
        max_z=MAX_Z,
        optimise_GP=True,
        traindir=TRAINDIR,
        data_corr=True,
        sdss="dr14",
    )

params = np.loadtxt(PARAMS_FILE)
params_gp, params_full = _split_parameter_vectors(params)
print(f"Loaded params from {PARAMS_FILE}")
print(f"Using {params_gp.size} GP params (+{params_full.size - params_gp.size} nuisance)")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    okf, predicted, std = lik.get_predicted(params_gp)

if lik.data_corr and params_full.size >= 14:
    for bb, z in enumerate(lik.zout):
        corr = lik.get_data_correction(okf[bb], params_full, z)
        predicted[bb] = predicted[bb] * corr

signal_vec = []
for bb, z in enumerate(lik.zout):
    ratio = T_fn(okf[bb], z, epsilon=1.0) - 1.0
    signal_vec.append(predicted[bb] * ratio)
signal = np.concatenate(signal_vec)

cov_blocks = []
for bb in range(len(lik.zout)):
    c_bb = lik.get_BOSS_error(bb)
    idp = np.where(lik.kf >= okf[bb][0])[0]
    cov_blocks.append(c_bb[np.ix_(idp, idp)])
c_full = scipy.linalg.block_diag(*cov_blocks)
c_inv = np.linalg.inv(c_full)

sn_sq = float(signal @ c_inv @ signal)
sn = np.sqrt(sn_sq)
n_needed = (2.0 / sn) ** 2 if sn > 0 else np.inf

sn_per_bin = []
offset = 0
for bb in range(len(lik.zout)):
    n = len(signal_vec[bb])
    s = signal[offset : offset + n]
    ci = c_inv[offset : offset + n, offset : offset + n]
    sn_per_bin.append(np.sqrt(float(s @ ci @ s)))
    offset += n
sn_per_bin = np.array(sn_per_bin)

print(f"\n(S/N)^2 = {sn_sq:.6f}")
print(f"S/N     = {sn:.6f}")
print(f"Need Nx = {n_needed:.2f}x BOSS DR14")

out_txt = os.path.join(OUTDIR, "fisher_chains_bestfit_allz.txt")
with open(out_txt, "w") as f:
    f.write("Full P1D Fisher forecast using bestfit_from_chains\n")
    f.write("=" * 56 + "\n")
    f.write(f"Params file: {PARAMS_FILE}\n")
    f.write(f"min_z={MIN_Z}, max_z={MAX_Z}\n")
    f.write("Signal: P_bestfit * (T-1)\n\n")
    f.write(f"(S/N)^2 = {sn_sq:.6f}\n")
    f.write(f"S/N     = {sn:.6f}\n")
    f.write(f"Need Nx = {n_needed:.2f}x BOSS DR14\n\n")
    f.write("Per-bin S/N:\n")
    for z, snz in zip(lik.zout, sn_per_bin):
        f.write(f"  z={z:.3f}  S/N={snz:.6f}\n")

print(f"Saved results to: {out_txt}")
print("DONE")
