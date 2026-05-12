"""
dde_degeneracy_scan_desi.py

DESI-EDRP version of the DDE degeneracy scan.

We keep the same finite-difference response method used in
dde_degeneracy_scan.py, but evaluate degeneracies in the DESI data space:
  - DESI k/z sampling
  - DESI covariance matrix

Because the current emulator supports z >= 2.2, this analysis uses the
DESI overlap region z in [2.2, 4.6].
"""

import csv
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dde_analysis.template import make_template_interpolator
from lyaemu.likelihood import LikelihoodClass


BASEDIR = "/Users/helenagescu/lya_emulator/dtau-48-48"
TRAINDIR = "/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf"
DATA_FILE = (
    "/Users/helenagescu/lya_emulator/lyaemu/data/desi_edrp_qmle_data/"
    "desi-edrp-lyasb1subt-p1d-detailed-results.txt"
)
COV_FILE = (
    "/Users/helenagescu/lya_emulator/lyaemu/data/desi_edrp_qmle_data/"
    "desi-edrp-lyasb1subt-cov-total-offdiag-results.txt"
)

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "dde_degeneracy_desi"
)
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    {"label": "all_z", "min_z": 2.2, "drop_z": [], "params_file": "bestfit_z2.2-4.6.txt"},
    {"label": "no_z2p2", "min_z": 2.2, "drop_z": [2.2], "params_file": "bestfit_z2.6-4.6.txt"},
    {
        "label": "no_z2p2_no_z2p4",
        "min_z": 2.2,
        "drop_z": [2.2, 2.4],
        "params_file": "bestfit_z2.6-4.6.txt",
    },
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


def _finite_diff_predictions(lik, p0_gp, p0_full, i, lo, hi, dp):
    p_plus = p0_gp.copy()
    p_minus = p0_gp.copy()
    mode = "central"
    if p0_gp[i] + dp > hi:
        mode = "backward"
    if p0_gp[i] - dp < lo:
        mode = "forward"
    if mode == "forward" and p0_gp[i] + dp > hi:
        return None
    if mode == "backward" and p0_gp[i] - dp < lo:
        return None

    if mode == "central":
        p_plus[i] += dp
        p_minus[i] -= dp
        denom = 2.0 * dp
    elif mode == "forward":
        p_plus[i] += dp
        denom = dp
    else:
        p_minus[i] -= dp
        denom = dp

    full_plus = p0_full.copy()
    full_minus = p0_full.copy()
    full_plus[: p0_gp.size] = p_plus
    full_minus[: p0_gp.size] = p_minus

    okf, pred_plus = _predict_with_corrections(lik, p_plus, full_plus)
    _, pred_minus = _predict_with_corrections(lik, p_minus, full_minus)
    return mode, denom, okf, pred_plus, pred_minus


def _interpolate_to_desi(okf_by_z, pred_by_z, z_sel, k_sel):
    out = np.empty_like(k_sel, dtype=float)
    for j, (zj, kj) in enumerate(zip(z_sel, k_sel)):
        z_key = round(float(zj), 1)
        if z_key not in okf_by_z:
            out[j] = np.nan
            continue
        k_native = okf_by_z[z_key]
        p_native = pred_by_z[z_key]
        if kj < k_native.min() or kj > k_native.max():
            out[j] = np.nan
            continue
        out[j] = np.interp(kj, k_native, p_native)
    return out


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


def run_config(cfg, z_all, k_all, cov_all):
    label = cfg["label"]
    pfile = os.path.join("dde_analysis/results", cfg["params_file"])

    print("=" * 72)
    print(f"DESI degeneracy scan: {label}")
    print("=" * 72)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    p0_full = np.loadtxt(pfile)
    p0_gp, p0_full = _split_parameter_vectors(p0_full)
    pnames_all = [nm for nm, _ in lik.get_pnames()]
    pnames_gp = pnames_all[: p0_gp.size]
    lims_gp = lik.param_limits[: p0_gp.size]

    # DESI rows in emulator overlap z range and optional drops
    mask = (z_all >= cfg["min_z"]) & (z_all <= 4.6 + 1e-6)
    for z_drop in cfg["drop_z"]:
        mask &= ~np.isclose(z_all, z_drop, atol=1e-6)
    idx = np.where(mask)[0]

    z_sel = z_all[idx]
    k_sel = k_all[idx]

    okf0, pred0_blocks = _predict_with_corrections(lik, p0_gp, p0_full)
    okf_by_z = {round(float(z), 1): np.array(okf0[bb], copy=True) for bb, z in enumerate(lik.zout)}
    pred0_by_z = {round(float(z), 1): np.array(pred0_blocks[bb], copy=True) for bb, z in enumerate(lik.zout)}
    p0_interp = _interpolate_to_desi(okf_by_z, pred0_by_z, z_sel, k_sel)
    valid = np.isfinite(p0_interp)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("No DESI points overlap emulator support for this configuration.")

    # Restrict to DESI points that are inside emulator interpolation support.
    idx = idx[valid]
    z_sel = z_sel[valid]
    k_sel = k_sel[valid]
    p0_interp = p0_interp[valid]
    c_sel = cov_all[np.ix_(idx, idx)]
    c_inv = np.linalg.inv(c_sel)

    dde_ratio = T_fn(k_sel, z_sel, epsilon=1.0) - 1.0
    dde_signal = p0_interp * dde_ratio

    z_unique = np.unique(z_sel)
    per_z = np.full((p0_gp.size, len(z_unique)), np.nan, dtype=float)
    rows = []

    for i, pname in enumerate(pnames_gp):
        lo, hi = lims_gp[i]
        dp = STEP_FRAC * (hi - lo)
        if dp <= 0:
            rows.append((pname, np.nan, np.nan, np.nan, np.nan, "invalid", 0.0))
            continue

        fd = _finite_diff_predictions(lik, p0_gp, p0_full, i, lo, hi, dp)
        if fd is None:
            rows.append((pname, np.nan, np.nan, np.nan, np.nan, "invalid", dp))
            continue
        mode, denom, okf, pred_plus, pred_minus = fd

        okf_by_z_fd = {round(float(z), 1): np.array(okf[bb], copy=True) for bb, z in enumerate(lik.zout)}
        plus_by_z = {round(float(z), 1): np.array(pred_plus[bb], copy=True) for bb, z in enumerate(lik.zout)}
        minus_by_z = {round(float(z), 1): np.array(pred_minus[bb], copy=True) for bb, z in enumerate(lik.zout)}
        p_plus_interp = _interpolate_to_desi(okf_by_z_fd, plus_by_z, z_sel, k_sel)
        p_minus_interp = _interpolate_to_desi(okf_by_z_fd, minus_by_z, z_sel, k_sel)

        if not np.all(np.isfinite(p_plus_interp)) or not np.all(np.isfinite(p_minus_interp)):
            rows.append((pname, np.nan, np.nan, np.nan, np.nan, "invalid", dp))
            continue

        dln = (p_plus_interp - p_minus_interp) / np.maximum(np.abs(0.5 * (p_plus_interp + p_minus_interp)), TINY)
        dln /= denom
        signal_i = p0_interp * dln

        cos_ratio = _cosine_similarity(dde_ratio, dln)
        pearson = float(np.corrcoef(dde_ratio, dln)[0, 1])
        cos_w = _weighted_cosine(dde_signal, signal_i, c_inv)
        sn_proj = float(dde_signal @ c_inv @ signal_i)
        rows.append((pname, cos_ratio, cos_w, pearson, sn_proj, mode, dp))

        for iz, zz in enumerate(z_unique):
            mz = np.isclose(z_sel, zz, atol=1e-6)
            per_z[i, iz] = _cosine_similarity(dde_ratio[mz], dln[mz])

    rows_sorted = sorted(rows, key=lambda r: np.nan_to_num(np.abs(r[2]), nan=-np.inf), reverse=True)

    txt_path = os.path.join(OUTDIR, f"degeneracy_rank_desi_{label}.txt")
    csv_path = os.path.join(OUTDIR, f"degeneracy_rank_desi_{label}.csv")
    with open(txt_path, "w") as f:
        f.write(f"DDE degeneracy ranking (DESI EDRP weighting) — {label}\n")
        f.write("=" * 80 + "\n")
        f.write(f"params file: {cfg['params_file']}\n")
        f.write(f"STEP_FRAC: {STEP_FRAC}\n")
        f.write(f"DESI z bins used: {z_unique}\n")
        f.write(f"n DESI points: {len(idx)}\n")
        f.write("emulator z support: [2.2, 4.6]\n\n")
        f.write(
            f"{'parameter':18s} {'cos_ratio':>12s} {'cos_weighted':>14s} "
            f"{'pearson':>10s} {'sn_proj':>12s} {'mode':>10s} {'dp':>11s}\n"
        )
        f.write("-" * 98 + "\n")
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

    bar_path = os.path.join(OUTDIR, f"degeneracy_cosine_bar_desi_{label}.png")
    heat_path = os.path.join(OUTDIR, f"degeneracy_perz_heatmap_desi_{label}.png")
    _make_barplot(
        [r[0] for r in rows],
        [r[2] for r in rows],
        title=f"DDE degeneracy ranking (DESI weighting, {label})\nCovariance-weighted cosine",
        outpath=bar_path,
    )
    _make_heatmap(
        zvals=z_unique,
        param_names=pnames_gp,
        mat=per_z,
        title=f"DDE vs emulator-parameter overlap by z (DESI, {label})",
        outpath=heat_path,
    )

    print(f"n DESI points used: {len(idx)}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {bar_path}")
    print(f"Saved: {heat_path}")
    print("Top weighted-degeneracy parameters:")
    for r in rows_sorted[:5]:
        print(f"  {r[0]:16s}  cos_w={r[2]: .4f}  cos_ratio={r[1]: .4f}  mode={r[5]}")
    print()


def main():
    arr = np.loadtxt(DATA_FILE)
    cov = np.loadtxt(COV_FILE)
    z_all = arr[:, 0]
    k_all = arr[:, 3]

    print("=" * 80)
    print("DDE degeneracy scan in DESI EDRP data space")
    print("=" * 80)
    print("Using emulator finite differences + DESI covariance weighting.")
    print("NOTE: restricted to emulator-supported redshifts z >= 2.2.")
    print()

    for cfg in CONFIGS:
        run_config(cfg, z_all, k_all, cov)

    print(f"All outputs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
