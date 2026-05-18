"""
Run Cobaya MCMC with epsilon fixed to 0 (LCDM baseline chain).

Usage:
    python dde_analysis/run_cobaya_lcdm_epsilon0_mcmc.py
"""

import os
import sys
from cobaya.run import run as cobaya_run
import numpy as np

# Ensure repo root is importable so Cobaya can locate dde_analysis.*
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from lyaemu.likelihood import LikelihoodClass


BASEDIR = os.environ.get("LYA_BASEDIR", os.path.join(REPO_ROOT, "dtau-48-48"))
TRAINDIR = os.environ.get("LYA_TRAINDIR", os.path.join(BASEDIR, "trained_mf"))
OUTDIR = os.environ.get(
    "COBAYA_OUTDIR_LCDM",
    os.path.join(REPO_ROOT, "dde_analysis", "results", "chains_lcdm_epsilon0"),
)
COVMAT_LCDM = os.environ.get(
    "COVMAT_LCDM",
    os.path.join(REPO_ROOT, "dde_analysis", "results", "covmats", "simeon_lcdm_14p.covmat"),
)
os.makedirs(OUTDIR, exist_ok=True)


def build_info():
    # Tunable runtime options (override via environment variables)
    burn_in = int(os.environ.get("COBAYA_BURN_IN", "1000"))
    max_samples = int(os.environ.get("COBAYA_MAX_SAMPLES", "30000"))
    rminus1_stop = float(os.environ.get("COBAYA_RMINUS1_STOP", "0.02"))
    proposal_div = float(os.environ.get("COBAYA_PROPOSAL_DIV", "300.0"))
    use_covmat = os.environ.get("USE_SIMEON_COVMAT", "0") == "1"
    output_suffix = os.environ.get("COBAYA_OUTPUT_SUFFIX", "pilot")

    lik = LikelihoodClass(
        basedir=BASEDIR,
        mean_flux="s",
        min_z=2.2,
        max_z=4.6,
        optimise_GP=True,
        traindir=TRAINDIR,
        data_corr=True,
        sdss="dr14",
    )
    pnames = lik.get_pnames()
    prange = lik.param_limits[:, 1] - lik.param_limits[:, 0]
    param_names = [p[0] for p in pnames]

    params = {
        pnames[i][0]: {
            "prior": {"min": float(lik.param_limits[i, 0]), "max": float(lik.param_limits[i, 1])},
            "proposal": float(prange[i] / proposal_div),
            "latex": pnames[i][1],
        }
        for i in range(lik.ndim)
    }

    # Start near known good point if available.
    bestfit_file = os.path.join(REPO_ROOT, "dde_analysis", "results", "bestfit_from_chains.txt")
    if os.path.exists(bestfit_file):
        bf = np.loadtxt(bestfit_file).reshape(-1)
        if bf.size >= len(param_names):
            for i, name in enumerate(param_names):
                params[name]["ref"] = float(bf[i])

    mcmc_cfg = {
        "burn_in": burn_in,
        "max_samples": max_samples,
        "Rminus1_stop": rminus1_stop,
        "learn_proposal": True,
        "output_every": "60s",
    }
    if use_covmat:
        mcmc_cfg["covmat"] = COVMAT_LCDM

    info = {
        "likelihood": {
            "dde_analysis.dde_cobaya_likelihood.CobayaDDELikelihoodClass": {
                "basedir": BASEDIR,
                "mean_flux": "s",
                "min_z": 2.2,
                "max_z": 4.6,
                "optimise_GP": True,
                "traindir": TRAINDIR,
                "data_corr": True,
                "sdss": "dr14",
                "include_emu": True,
                "hprior": False,
                "oprior": False,
                "bhprior": False,
                "epsilon": 0.0,
            }
        },
        "params": params,
        "sampler": {"mcmc": mcmc_cfg},
        "output": os.path.join(OUTDIR, f"lcdm_epsilon0_{output_suffix}"),
    }
    return info


def main():
    info = build_info()
    print("Starting Cobaya MCMC with epsilon fixed to 0 (LCDM)...")
    print(f"Output prefix: {info['output']}")
    print(f"Sampler config: {info['sampler']['mcmc']}")
    updated_info, sampler = cobaya_run(info, resume=True)
    print("Done.")
    return updated_info, sampler


if __name__ == "__main__":
    main()

