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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu.likelihood import LikelihoodClass


BASEDIR = "/Users/helenagescu/lya_emulator/dtau-48-48"
TRAINDIR = "/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf"
OUTDIR = "/Users/helenagescu/lya_emulator/dde_analysis/results/chains_dde_epsilon"
COVMAT_DDE = "/Users/helenagescu/lya_emulator/dde_analysis/results/covmats/simeon_plus_epsilon_15p.covmat"
os.makedirs(OUTDIR, exist_ok=True)


def build_info():
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
        "sampler": {
            "mcmc": {
                "burn_in": 20000,
                "max_samples": 80000,
                "Rminus1_stop": 0.01,
                "learn_proposal": True,
                "covmat": COVMAT_DDE,
                "output_every": "60s",
            }
        },
        "output": os.path.join(OUTDIR, "dde_epsilon"),
    }
    return info


def main():
    info = build_info()
    print("Starting Cobaya MCMC with sampled epsilon...")
    print(f"Output prefix: {info['output']}")
    print(f"Proposal covmat: {COVMAT_DDE}")
    updated_info, sampler = cobaya_run(info, resume=True)
    print("Done.")
    return updated_info, sampler


if __name__ == "__main__":
    main()

