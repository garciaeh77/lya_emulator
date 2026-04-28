"""
find_bestfit_restart.py
Restarts optimisation from the previously found best-fit point.
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

# Load previous best-fit as starting point
p0 = np.loadtxt('dde_analysis/results/bestfit_params.txt')
print(f"Starting from previous best-fit:")
print(f"  params = {p0}")

best_chi2 = [1233.43]  # track improvement

def neg_loglike(params):
    if np.any(params < lik.param_limits[:, 0]) or \
       np.any(params > lik.param_limits[:, 1]):
        return 1e10
    ll = lik.likelihood(params, include_emu=True,
                        data_power=None,
                        hprior=False, oprior=False)
    chi2 = -2 * ll
    if chi2 < best_chi2[0]:
        best_chi2[0] = chi2
        print(f"  New best chi2 = {chi2:.2f}  params = {np.round(params, 4)}",
              flush=True)
    return -ll

# Try multiple restarts from small perturbations around current best
print("\nRunning restart optimisation...")
best_result = None
best_fun = 1e10

for trial in range(3):
    # Add small random perturbation to escape local minimum
    np.random.seed(trial)
    perturbation = np.random.randn(len(p0)) * 0.01 * (
        lik.param_limits[:, 1] - lik.param_limits[:, 0]
    )
    p_start = np.clip(
        p0 + perturbation,
        lik.param_limits[:, 0],
        lik.param_limits[:, 1]
    )
    print(f"\nTrial {trial+1}/3 ...")
    result = scipy.optimize.minimize(
        neg_loglike, p_start,
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-4,
                 'fatol': 0.5, 'adaptive': True, 'disp': True}
    )
    if result.fun < best_fun:
        best_fun = result.fun
        best_result = result
        print(f"  Trial {trial+1} improved: chi2 = {2*result.fun:.2f}")

print(f"\nBest chi2 across all trials = {2*best_result.fun:.4f}")
print(f"Best params = {best_result.x}")

# Save
outfile = 'dde_analysis/results/bestfit_params_restarted.txt'
np.savetxt(outfile, best_result.x.reshape(1, -1),
           header=' '.join([n for n, _ in lik.get_pnames()]))
print(f"Saved to {outfile}")
