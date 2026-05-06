"""
plot_fernandez2023_fig5_union3.py

Reproduce the Fernandez+2023 Figure-5 style P1D comparison panels
and add an explicit Union3 DDE prediction overlay.

This script uses:
  - BOSS DR14 data vector + covariance errors
  - LCDM best-fit parameters from a text file
  - DDE-injected prediction using DDELikelihood(epsilon=1)

Output:
  dde_analysis/results/fig5_union3/fernandez2023_fig5_union3.png
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyaemu import lyman_data as ld
from lyaemu import likelihood as lk
from dde_analysis.dde_likelihood import DDELikelihood

# Colors close to the original style
C_MIDNIGHT = "#0E2240"
C_SKYLINE = "#1D428A"
C_FLATIRONS = "#8B2131"

BASEDIR = "/Users/helenagescu/lya_emulator/dtau-48-48"
TRAINDIR = "/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf"
PARAMS_FILE = "/Users/helenagescu/lya_emulator/dde_analysis/results/bestfit_from_chains.txt"

OUTDIR = "/Users/helenagescu/lya_emulator/dde_analysis/results/fig5_union3"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, "fernandez2023_fig5_union3_labeled.png")


def split_params(full_params):
    """Use first 11 for GP emulator and full 14 for data corrections."""
    p = np.asarray(full_params).reshape(-1)
    if p.size < 11:
        raise ValueError(f"Expected >=11 parameters, got {p.size}")
    return p[:11], p


def to_kpfpi(okf_list, pf_list):
    """Convert P_F(k) to k P_F / pi for each z bin."""
    out = []
    for k, p in zip(okf_list, pf_list):
        out.append(k * p / np.pi)
    return out


def main():
    # BOSS DR14 observations
    boss = ld.BOSSData()
    boss_k = boss.kf.reshape(13, -1)  # ordered low->high z in source
    boss_pf = boss.get_pf().reshape(13, -1)[::-1]  # reverse to match lik.zout
    boss_var = boss.covar_diag.reshape(13, -1)[::-1]
    boss_pf_plot = boss_pf * boss_k / np.pi
    boss_err_plot = np.sqrt(boss_var * boss_k**2 / np.pi**2)

    params = np.loadtxt(PARAMS_FILE)
    params_gp, params_full = split_params(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lik_lcdm = lk.LikelihoodClass(
            BASEDIR,
            mean_flux="s",
            min_z=2.2,
            max_z=4.6,
            traindir=TRAINDIR,
            data_corr=True,
            optimise_GP=True,
            sdss="dr14",
        )
        lik_dde = DDELikelihood(
            BASEDIR,
            mean_flux="s",
            min_z=2.2,
            max_z=4.6,
            traindir=TRAINDIR,
            data_corr=True,
            optimise_GP=False,
            sdss="dr14",
            epsilon=1.0,
        )

    # Reuse trained GP from baseline likelihood for speed/consistency
    lik_dde.gpemu = lik_lcdm.gpemu

    okf_lcdm, pred_lcdm, _ = lik_lcdm.get_predicted(params_gp)
    okf_dde, pred_dde, _ = lik_dde.get_predicted(params_full)

    # Apply nuisance data corrections (DLA+SiIII) in both models
    for bb, z in enumerate(lik_lcdm.zout):
        corr_lcdm = lik_lcdm.get_data_correction(okf_lcdm[bb], params_full, z)
        corr_dde = lik_dde.get_data_correction(okf_dde[bb], params_full, z)
        pred_lcdm[bb] *= corr_lcdm
        pred_dde[bb] *= corr_dde

    pred_lcdm_plot = to_kpfpi(okf_lcdm, pred_lcdm)
    pred_dde_plot = to_kpfpi(okf_dde, pred_dde)

    # Figure-5 style panel grouping
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(
        figsize=(10.625 * 2, 11 * 1.75),
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )
    axes = axes.flatten()
    zz = np.round(lik_lcdm.zout, 1)

    for mm, ax in enumerate(axes):
        mplot = [2 * mm, 2 * mm + 1]
        if mm == 5:
            mplot.append(2 * mm + 2)

        for m in mplot:
            ax.plot(
                boss_k[m], boss_pf_plot[m], "-o",
                color=C_MIDNIGHT, lw=1.8, ms=3.5, zorder=0
            )
            ax.fill_between(
                boss_k[m],
                boss_pf_plot[m] - boss_err_plot[m],
                boss_pf_plot[m] + boss_err_plot[m],
                color=C_MIDNIGHT,
                alpha=0.28,
                zorder=0,
            )
            ax.plot(okf_lcdm[m], pred_lcdm_plot[m], "--", color=C_SKYLINE, lw=2.4, zorder=2)
            ax.plot(okf_dde[m], pred_dde_plot[m], "-", color=C_FLATIRONS, lw=2.4, zorder=3)

        ymin = 1.35 * np.min(boss_pf_plot[mplot[-1]])
        ax.text(0.014, ymin, f"z: {zz[np.min(mplot)]}-{zz[np.max(mplot)]}", fontsize=18)
        ax.tick_params(which="both", direction="inout", length=9, labelsize=12)

    # Labels in first panel (keep original style)
    y0 = np.max(boss_pf_plot[0] + boss_err_plot[0]) * 0.94
    axes[0].text(0.0015, y0, "Chabanier 2019 data", fontsize=14, color=C_MIDNIGHT)
    axes[0].text(0.0015, y0 - 0.06 * y0, "LCDM fit (blue dashed)", fontsize=14, color=C_SKYLINE)
    axes[0].text(0.0015, y0 - 0.12 * y0, "DDE Union3 (red solid, epsilon=1)", fontsize=14, color=C_FLATIRONS)
    axes[0].text(
        0.0015,
        y0 - 0.18 * y0,
        "Top-z bins: red/blue overlap nearly exactly",
        fontsize=12,
        color=C_MIDNIGHT,
    )

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel(r"$k P_F(k) / \pi$", size=24, labelpad=20)
    plt.xlabel("k [s/km]", size=24)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(OUTFILE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
