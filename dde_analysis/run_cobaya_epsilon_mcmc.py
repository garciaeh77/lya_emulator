"""
Run full Cobaya MCMC with sampled DDE template amplitude epsilon.

Usage:
    python dde_analysis/run_cobaya_epsilon_mcmc.py
"""

import os
import sys
import numpy as np
from cobaya.run import run as cobaya_run

# Ensure repo root is importable so Cobaya can locate dde_analysis.*
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from lyaemu.likelihood import LikelihoodClass


BASEDIR = os.environ.get("LYA_BASEDIR", os.path.join(REPO_ROOT, "dtau-48-48"))
TRAINDIR = os.environ.get("LYA_TRAINDIR", os.path.join(BASEDIR, "trained_mf"))
OUTDIR = os.environ.get(
    "COBAYA_OUTDIR_DDE",
    os.path.join(REPO_ROOT, "dde_analysis", "results", "chains_dde_epsilon"),
)
COVMAT_DDE = os.environ.get(
    "COVMAT_DDE",
    os.path.join(REPO_ROOT, "dde_analysis", "results", "covmats", "simeon_plus_epsilon_15p.covmat"),
)
os.makedirs(OUTDIR, exist_ok=True)


def build_info():
    use_covmat = os.environ.get("USE_DDE_COVMAT", "0") == "1"

    # Use baseline likelihood only to get parameter names/limits.
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

    params = {
        pnames[i][0]: {
            "prior": {"min": float(lik.param_limits[i, 0]), "max": float(lik.param_limits[i, 1])},
            "proposal": float(prange[i] / 80.0),
            "latex": pnames[i][1],
        }
        for i in range(lik.ndim)
    }

    # Add sampled epsilon parameter.
    params["epsilon"] = {
        "prior": {"min": 0.0, "max": 2.0},
        "proposal": 0.03,
        "latex": r"\epsilon",
    }

    mcmc_cfg = {
        "burn_in": 20000,
        "max_samples": 80000,
        "Rminus1_stop": 0.01,
        "learn_proposal": True,
        "output_every": "60s",
    }
    if use_covmat:
        mcmc_cfg["covmat"] = COVMAT_DDE

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
        "output": os.path.join(OUTDIR, "dde_epsilon"),
    }
    return info


def main():
    info = build_info()
    print("Starting Cobaya MCMC with sampled epsilon...")
    print(f"Output prefix: {info['output']}")
    if "covmat" in info["sampler"]["mcmc"]:
        print(f"Proposal covmat: {info['sampler']['mcmc']['covmat']}")
    else:
        print("Proposal covmat: disabled (learning proposal from scratch)")
    updated_info, sampler = cobaya_run(info, resume=True)
    print("Done.")
    return updated_info, sampler


if __name__ == "__main__":
    main()

