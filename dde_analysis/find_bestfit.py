"""
find_bestfit.py
Finds best-fit parameters by minimising -log-likelihood using scipy.
Run this while waiting for Simeon to share the official best-fit values.
"""
import numpy as np
import scipy.optimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lyaemu.likelihood import LikelihoodClass

BASEDIR  = '/Users/helenagescu/lya_emulator/dtau-48-48'
TRAINDIR = '/Users/helenagescu/lya_emulator/dtau-48-48/trained_mf'

lik = LikelihoodClass(
    basedir=BASEDIR, mean_flux='s',
    min_z=2.2, max_z=4.6,
    optimise_GP=True, traindir=TRAINDIR,
    data_corr=True, sdss='dr14',
)

# Starting point: midpoint of parameter ranges
p0 = (lik.param_limits[:, 0] + lik.param_limits[:, 1]) / 2.

def neg_loglike(params):
    # Check bounds
    if np.any(params < lik.param_limits[:, 0]) or \
       np.any(params > lik.param_limits[:, 1]):
        return 1e10
    ll = lik.likelihood(params, include_emu=True,
                        data_power=None,
                        hprior=False, oprior=False)
    print(f"  chi2 = {-2*ll:.2f}", flush=True)
    return -ll

print("Starting optimisation...")
result = scipy.optimize.minimize(
    neg_loglike, p0,
    method='Nelder-Mead',
    options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1.0,
             'adaptive': True, 'disp': True}
)

print(f"\nOptimisation complete.")
print(f"Best chi2     = {2*result.fun:.4f}")
print(f"Best params   = {result.x}")

# Save
np.savetxt(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'results', 'bestfit_params.txt'),
    result.x.reshape(1, -1),
    header=' '.join([n for n, _ in lik.get_pnames()])
)
print("Saved to dde_analysis/results/bestfit_params.txt")
