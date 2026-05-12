"""
dde_degeneracy_scan.py

Estimate DDE-template degeneracies with emulator parameters by comparing:
  - R_DDE(k,z) = T_union3(k,z,epsilon=1) - 1
  - R_i(k,z)   = d ln P(k,z) / d p_i  (finite differencing around fiducial)

Outputs:
  - Ranked degeneracy tables (txt + csv)
  - Heatmap of per-z cosine similarities
  - Bar chart of global cosine similarities
"""

import csv
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
    os.path.dirname(os.path.abspath(__file__)), "results", "dde_degeneracy"
)
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    {"label": "all_z", "min_z": 2.2, "params_file": "bestfit_z2.2-4.6.txt"},
    {"label": "no_z2p2", "min_z": 2.4, "params_file": "bestfit_z2.6-4.6.txt"},
    {"label": "no_z2p2_no_z2p4", "min_z": 2.6, "params_file": "bestfit_z2.6-4.6.txt"},
]

STEP_FRAC = 0.01
TINY = 1e-15

T_fn = make_template_interpolator()


def _split_parameter_vectors(full_params):
    p = np.asarray(full_params).reshape(-1)
    if p.size < 11:
        raise ValueError(f"Expected at least 11 parameters, got {p.size}: {p}")
    return p[:11].copy(), p.copy()


def _safe_norm(vec):
    return float(np.sqrt(np.dot(vec, vec)))


def _cosine_similarity(a, b):
    an = _safe_norm(a)
    bn = _safe_norm(b)
    if an <= 0 or bn <= 0:
        return np.nan
    return float(np.dot(a, b) / (an * bn))


def _weighted_cosine(a, b, c_inv):
    aa = float(a @ c_inv @ a)
    bb = float(b @ c_inv @ b)
    if aa <= 0 or bb <= 0:
        return np.nan
    ab = float(a @ c_inv @ b)
    return ab / np.sqrt(aa * bb)


def _build_covariance_for_okf(lik, okf):
    cov_blocks = []
    for bb in range(len(lik.zout)):
        c_bb = lik.get_BOSS_error(bb)
        idp = np.where(lik.kf >= okf[bb][0])[0]
        cov_blocks.append(c_bb[np.ix_(idp, idp)])
    return scipy.linalg.block_diag(*cov_blocks)


def _predict_with_corrections(lik, gp_params, full_params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        okf, pred, _ = lik.get_predicted(gp_params)
    pred = [np.array(p, copy=True) for p in pred]
    if lik.data_corr and full_params.size >= 14:
        for bb, z in enumerate(lik.zout):
            corr = lik.get_data_correction(okf[bb], full_params, z)
            pred[bb] *= corr
    return okf, pred


def _flatten_blocks(blocks):
    return np.concatenate([np.asarray(b).reshape(-1) for b in blocks])


def _finite_diff_response(lik, p0_gp, p0_full, i, lo, hi, dp):
    p_plus = p0_gp.copy()
    p_minus = p0_gp.copy()
    mode = "central"
    denom = 2.0 * dp

    if p0_gp[i] + dp > hi:
        mode = "backward"
    if p0_gp[i] - dp < lo:
        mode = "forward"
    if mode == "forward" and p0_gp[i] + dp > hi:
        return None, "invalid"
    if mode == "backward" and p0_gp[i] - dp < lo:
        return None, "invalid"

    if mode == "central":
        p_plus[i] = p0_gp[i] + dp
        p_minus[i] = p0_gp[i] - dp
    elif mode == "forward":
        p_plus[i] = p0_gp[i] + dp
        p_minus[i] = p0_gp[i]
        denom = dp
    else:
        p_plus[i] = p0_gp[i]
        p_minus[i] = p0_gp[i] - dp
        denom = dp

    full_plus = p0_full.copy()
    full_minus = p0_full.copy()
    full_plus[: p0_gp.size] = p_plus
    full_minus[: p0_gp.size] = p_minus

    _, pred_plus = _predict_with_corrections(lik, p_plus, full_plus)
    _, pred_minus = _predict_with_corrections(lik, p_minus, full_minus)

    dln_blocks = []
    for b_plus, b_minus in zip(pred_plus, pred_minus):
        # d ln P / dp_i
        dln = (b_plus - b_minus) / np.maximum(np.abs(b_plus + b_minus) * 0.5, TINY)
        dln /= denom
        dln_blocks.append(dln)
    return dln_blocks, mode


def _make_barplot(param_names, values, title, outpath):
    order = np.argsort(np.nan_to_num(np.abs(values), nan=-np.inf))[::-1]
    names_ord = [param_names[i] for i in order]
    vals_ord = np.array(values)[order]
    colors = ["tab:red" if np.isfinite(v) and v < 0 else "tab:blue" for v in vals_ord]

    fig_h = max(4.0, 0.35 * len(names_ord))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y = np.arange(len(names_ord))
    ax.barh(y, vals_ord, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names_ord, fontsize=9)
    ax.axvline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("Cosine similarity with DDE ratio")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_heatmap(zvals, param_names, mat, title, outpath):
    fig_h = max(4.0, 0.35 * len(param_names))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(zvals)))
    ax.set_xticklabels([f"{z:.1f}" for z in zvals])
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=8)
    ax.set_xlabel("Redshift bin")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Per-z cosine similarity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_config(cfg):
    label = cfg["label"]
    min_z = cfg["min_z"]
    pfile = os.path.join("dde_analysis/results", cfg["params_file"])

    print("=" * 72)
    print(f"Degeneracy scan: {label} (min_z={min_z})")
    print("=" * 72)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lik = LikelihoodClass(
            basedir=BASEDIR,
            mean_flux="s",
            min_z=min_z,
            max_z=4.6,
            optimise_GP=True,
            traindir=TRAINDIR,
            data_corr=True,
            sdss="dr14",
        )

    p0_full = np.loadtxt(pfile)
    p0_gp, p0_full = _split_parameter_vectors(p0_full)
    pnames_all = [nm for nm, _ in lik.get_pnames()]
    pnames_gp = pnames_all[: p0_gp.size]
    lims_gp = lik.param_limits[: p0_gp.size]

    okf, pred0_blocks = _predict_with_corrections(lik, p0_gp, p0_full)
    p0_flat = _flatten_blocks(pred0_blocks)
    dde_ratio_blocks = [T_fn(okf[bb], z, epsilon=1.0) - 1.0 for bb, z in enumerate(lik.zout)]
    dde_ratio_flat = _flatten_blocks(dde_ratio_blocks)
    dde_signal_flat = p0_flat * dde_ratio_flat

    c_full = _build_covariance_for_okf(lik, okf)
    c_inv = np.linalg.inv(c_full)

    print(f"Using {len(lik.zout)} redshift bins and {len(p0_flat)} total P1D points.")
    print(f"Template ratio range: [{dde_ratio_flat.min():.4f}, {dde_ratio_flat.max():.4f}]")

    rows = []
    per_z = np.full((p0_gp.size, len(lik.zout)), np.nan, dtype=float)

    for i, pname in enumerate(pnames_gp):
        lo, hi = lims_gp[i]
        rng = hi - lo
        dp = STEP_FRAC * rng
        if dp <= 0:
            rows.append((pname, np.nan, np.nan, np.nan, np.nan, "invalid", 0.0))
            continue

        dln_blocks, mode = _finite_diff_response(lik, p0_gp, p0_full, i, lo, hi, dp)
        if dln_blocks is None:
            rows.append((pname, np.nan, np.nan, np.nan, np.nan, "invalid", dp))
            continue

        dln_flat = _flatten_blocks(dln_blocks)
        signal_i_flat = p0_flat * dln_flat

        cos_ratio = _cosine_similarity(dde_ratio_flat, dln_flat)
        pearson = float(np.corrcoef(dde_ratio_flat, dln_flat)[0, 1])
        cos_w = _weighted_cosine(dde_signal_flat, signal_i_flat, c_inv)
        sn_proj = float(dde_signal_flat @ c_inv @ signal_i_flat)

        rows.append((pname, cos_ratio, cos_w, pearson, sn_proj, mode, dp))

        for bb in range(len(lik.zout)):
            per_z[i, bb] = _cosine_similarity(dde_ratio_blocks[bb], dln_blocks[bb])

    rows_sorted = sorted(rows, key=lambda r: np.nan_to_num(np.abs(r[2]), nan=-np.inf), reverse=True)

    txt_path = os.path.join(OUTDIR, f"degeneracy_rank_{label}.txt")
    csv_path = os.path.join(OUTDIR, f"degeneracy_rank_{label}.csv")
    with open(txt_path, "w") as f:
        f.write(f"DDE degeneracy ranking — {label}\n")
        f.write("=" * 72 + "\n")
        f.write(f"params file: {cfg['params_file']}\n")
        f.write(f"STEP_FRAC: {STEP_FRAC}\n")
        f.write(f"redshift bins: {np.array(lik.zout)}\n")
        f.write(f"n points: {len(p0_flat)}\n\n")
        f.write(
            f"{'parameter':18s} {'cos_ratio':>12s} {'cos_weighted':>14s} "
            f"{'pearson':>10s} {'sn_proj':>12s} {'mode':>10s} {'dp':>11s}\n"
        )
        f.write("-" * 95 + "\n")
        for r in rows_sorted:
            f.write(
                f"{r[0]:18s} {r[1]:12.5f} {r[2]:14.5f} {r[3]:10.5f} "
                f"{r[4]:12.5e} {r[5]:>10s} {r[6]:11.5e}\n"
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["parameter", "cos_ratio", "cos_weighted", "pearson", "sn_proj", "fd_mode", "delta_p"]
        )
        for r in rows_sorted:
            writer.writerow(r)

    bar_path = os.path.join(OUTDIR, f"degeneracy_cosine_bar_{label}.png")
    heat_path = os.path.join(OUTDIR, f"degeneracy_perz_heatmap_{label}.png")
    _make_barplot(
        [r[0] for r in rows],
        [r[2] for r in rows],
        title=f"DDE degeneracy ranking ({label})\nCovariance-weighted cosine",
        outpath=bar_path,
    )
    _make_heatmap(
        zvals=np.array(lik.zout),
        param_names=pnames_gp,
        mat=per_z,
        title=f"DDE vs emulator-parameter shape overlap by z ({label})",
        outpath=heat_path,
    )

    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {bar_path}")
    print(f"Saved: {heat_path}")
    top = rows_sorted[:5]
    print("Top weighted-degeneracy parameters:")
    for r in top:
        print(f"  {r[0]:16s}  cos_w={r[2]: .4f}  cos_ratio={r[1]: .4f}  mode={r[5]}")
    print()


def main():
    print("=" * 72)
    print("DDE degeneracy scan using finite-difference emulator responses")
    print("=" * 72)
    for cfg in CONFIGS:
        run_config(cfg)
    print(f"All outputs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
