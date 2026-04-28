"""
forward_model_dde.py

Evaluates how much the chi-squared changes when the DDE spectral tilt
from Garza et al. 2026 (arXiv:2601.00767) is injected into the
lya_emulator predicted flux power spectrum.

Usage:
    python forward_model_dde.py
"""

import numpy as np
import matplotlib.pyplot as plt
from lyaemu.likelihood import LikelihoodClass
from lyaemu import flux_power


# ─────────────────────────────────────────────────────────────────────────────
# 1. INSTANTIATE THE BASELINE LIKELIHOOD
# ─────────────────────────────────────────────────────────────────────────────

BASEDIR   = '/path/to/your/emulator'   # ← change this
TRAINDIR  = '/path/to/trained_gp'      # ← change this (or set to None)
MIN_Z     = 2.2
MAX_Z     = 4.6

lik = LikelihoodClass(
    basedir=BASEDIR,
    mean_flux='s',
    min_z=MIN_Z,
    max_z=MAX_Z,
    optimise_GP=True,
    traindir=TRAINDIR,
    data_corr=True,
    sdss='dr14',
)

# Print parameter names so you know the ordering
pnames = lik.get_pnames()
print("Parameter vector order:")
for i, (name, latex) in enumerate(pnames):
    lo, hi = lik.param_limits[i]
    print(f"  [{i:2d}] {name:20s}  range [{lo:.4f}, {hi:.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE BEST-FIT PARAMETERS
#    Replace these with your actual best-fit values.
#    The vector must cover ALL parameters including mean flux,
#    cosmological, thermal, and (if data_corr=True) SiIII + DLA params.
# ─────────────────────────────────────────────────────────────────────────────

# Convenience: start from the midpoint of each parameter range
params_bestfit = (lik.param_limits[:, 0] + lik.param_limits[:, 1]) / 2.

# ── OR load from a previous MCMC chain, e.g. ──────────────────────────────
# import getdist.mcsamples as gmc
# samples = gmc.loadMCSamples('/path/to/chain')
# params_bestfit = samples.getMeans()
# ──────────────────────────────────────────────────────────────────────────

print(f"\nBest-fit parameter vector:\n{params_bestfit}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE BASELINE CHI-SQUARED
# ─────────────────────────────────────────────────────────────────────────────

chi2_baseline = lik.likelihood(
    params_bestfit,
    include_emu=True,
    data_power=None,     # None → use BOSS DR14 data
    hprior=False,
    oprior=False,
)

print(f"\nBaseline log-likelihood  = {chi2_baseline:.4f}")
print(f"Baseline chi2            = {-2 * chi2_baseline:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD THE DDE SPECTRAL TILT TEMPLATE
#
#    This encodes the ratio [Δ²_DDE(k,z) / Δ²_ΛCDM(k,z)] - 1
#    read off from Figure 9 of Garza et al. 2026.
#
#    The template is parameterised as a power-law tilt in k,
#    with a redshift-dependent amplitude A(z) and a fixed
#    pivot scale k_pivot. The functional form is:
#
#        T(k, z) = 1 + A(z) * log10(k / k_pivot)
#
#    where A(z) < 0 (tilt is negative: more power at low k,
#    less at high k).
#
#    Values below are read from the Union3 model in Figure 9,
#    which gives the largest signal:
#        ~ +2% at k = 0.01 s/km  (large scales)
#        ~ -4% at k = 0.1  s/km  (small scales)
#    implying a slope of about -0.06 per decade in k.
#
#    The amplitude grows with decreasing redshift, becoming
#    most pronounced below z ~ 3.
#
#    IMPORTANT: replace these numbers with values read directly
#    from the paper's Figure 9 for the model you want to test.
#    The three available models are Pantheon, DESY5, Union3.
# ─────────────────────────────────────────────────────────────────────────────

# Pivot scale in s/km — approximately where the tilt crosses zero
K_PIVOT = 0.018   # s/km, from Figure 9 Pantheon model at z ~ 2.33

# Redshift-dependent tilt amplitude A(z).
# A(z) < 0 → more large-scale power, less small-scale power (the DDE tilt).
# These are rough values digitised from Figure 9 of Garza et al.
# for the Union3 model (largest signal). Adjust for Pantheon or DESY5.
# Format: {redshift: A(z)}
_A_Z_TABLE = {
    4.0:   0.000,   # no tilt at high z — just a normalization offset
    3.6:  -0.010,
    3.4:  -0.015,
    3.2:  -0.020,
    3.0:  -0.025,
    2.8:  -0.030,
    2.6:  -0.035,
    2.4:  -0.040,
    2.33: -0.045,
    2.2:  -0.050,
    1.491:-0.045,
    1.317:-0.040,
}

_z_table  = np.array(sorted(_A_Z_TABLE.keys()))
_A_table  = np.array([_A_Z_TABLE[z] for z in _z_table])

def dde_template(k_kms, z):
    """
    Returns the multiplicative DDE template T(k, z) such that
        P_DDE(k, z) = T(k, z) * P_LCDM(k, z)

    Parameters
    ----------
    k_kms : array_like
        Wavenumbers in s/km.
    z : float
        Redshift.

    Returns
    -------
    T : np.ndarray
        Multiplicative template, same shape as k_kms.
        Values > 1 at large scales, < 1 at small scales.
    """
    # Interpolate amplitude at this redshift
    A_z = np.interp(z, _z_table, _A_table)

    # Tilt: T(k) = 1 + A(z) * log10(k / k_pivot)
    # log10(k/k_pivot) is negative at large scales (k < k_pivot)
    # and positive at small scales (k > k_pivot).
    # With A(z) < 0 this gives T > 1 at large scales (correct)
    # and T < 1 at small scales (correct).
    log_ratio = np.log10(np.asarray(k_kms) / K_PIVOT)
    T = 1.0 + A_z * log_ratio
    return T


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUBCLASS LikelihoodClass TO INJECT THE TEMPLATE
#
#    We override get_predicted so that the emulator output is
#    multiplied by the DDE template before being returned.
#    Everything else — the covariance matrix, the SiIII/DLA
#    corrections, the chi2 calculation — stays identical.
# ─────────────────────────────────────────────────────────────────────────────

class DDELikelihood(LikelihoodClass):
    """
    Identical to LikelihoodClass except that get_predicted
    multiplies the emulator flux power by the DDE spectral tilt template.
    """

    def get_predicted(self, params):
        """
        Call the parent get_predicted, then multiply each redshift
        bin's flux power by the DDE template T(k, z).
        """
        # Get the baseline emulator prediction (line 282 equivalent)
        okf, predicted, std = super().get_predicted(params)

        # Apply the DDE template bin by bin
        for bb, z in enumerate(self.zout):
            T = dde_template(okf[bb], z)
            predicted[bb] = predicted[bb] * T
            # The emulator uncertainty scales the same way.
            # This is conservative — in reality the template is
            # deterministic so it adds no extra uncertainty.
            std[bb] = std[bb] * T

        return okf, predicted, std


# ─────────────────────────────────────────────────────────────────────────────
# 6. INSTANTIATE THE DDE LIKELIHOOD AND EVALUATE CHI-SQUARED
#
#    We reuse the already-trained GP from the baseline instance
#    by copying the relevant attributes, avoiding a second training run.
# ─────────────────────────────────────────────────────────────────────────────

lik_dde = DDELikelihood(
    basedir=BASEDIR,
    mean_flux='s',
    min_z=MIN_Z,
    max_z=MAX_Z,
    optimise_GP=False,   # ← do NOT retrain; we copy the GP below
    traindir=TRAINDIR,
    data_corr=True,
    sdss='dr14',
)

# Copy the trained GP from the baseline instance
lik_dde.gpemu   = lik.gpemu
lik_dde.icov_bin = lik.icov_bin
lik_dde.cdet    = lik.cdet

chi2_dde = lik_dde.likelihood(
    params_bestfit,
    include_emu=True,
    data_power=None,
    hprior=False,
    oprior=False,
)

print(f"\nDDE log-likelihood       = {chi2_dde:.4f}")
print(f"DDE chi2                 = {-2 * chi2_dde:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPUTE Δχ² AND SIGNIFICANCE
# ─────────────────────────────────────────────────────────────────────────────

delta_chi2 = -2 * chi2_dde - (-2 * chi2_baseline)
# delta_chi2 > 0 means DDE template makes the fit worse (BOSS data disfavours DDE)
# delta_chi2 < 0 means DDE template improves the fit (BOSS data prefers DDE)

# Approximate significance in sigma assuming chi2 ~ chi2(1 dof)
# (one extra degree of freedom = the template amplitude)
import scipy.stats
significance = np.sqrt(np.abs(delta_chi2))   # rough
p_value = scipy.stats.chi2.sf(np.abs(delta_chi2), df=1)

print(f"\n{'='*50}")
print(f"Δχ²                      = {delta_chi2:+.4f}")
print(f"Significance (1 dof)     ≈ {significance:.2f} σ")
print(f"p-value                  = {p_value:.4f}")
if delta_chi2 > 0:
    print("Interpretation: BOSS data DISFAVOURS the DDE tilt")
else:
    print("Interpretation: BOSS data PREFERS the DDE tilt")
print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIONAL: SCAN Δχ² AS A FUNCTION OF TEMPLATE AMPLITUDE
#
#    Instead of a fixed template, scale it by a free amplitude ε ∈ [0, 1]
#    where ε = 0 is ΛCDM and ε = 1 is the full Union3 DDE prediction.
#    This gives you a profile likelihood in the template amplitude.
# ─────────────────────────────────────────────────────────────────────────────

class ScaledDDELikelihood(LikelihoodClass):
    """DDE likelihood with a tunable template amplitude epsilon."""

    def __init__(self, epsilon, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def get_predicted(self, params):
        okf, predicted, std = super().get_predicted(params)
        for bb, z in enumerate(self.zout):
            # Blend between ΛCDM (epsilon=0) and full DDE (epsilon=1)
            T = 1.0 + self.epsilon * (dde_template(okf[bb], z) - 1.0)
            predicted[bb] *= T
            std[bb] *= T
        return okf, predicted, std


epsilon_grid = np.linspace(0, 2, 21)   # go beyond 1 to look for preferred value
delta_chi2_profile = []

for eps in epsilon_grid:
    lik_eps = ScaledDDELikelihood(
        epsilon=eps,
        basedir=BASEDIR,
        mean_flux='s',
        min_z=MIN_Z,
        max_z=MAX_Z,
        optimise_GP=False,
        traindir=TRAINDIR,
        data_corr=True,
        sdss='dr14',
    )
    lik_eps.gpemu    = lik.gpemu
    lik_eps.icov_bin = lik.icov_bin
    lik_eps.cdet     = lik.cdet

    ll = lik_eps.likelihood(params_bestfit, include_emu=True, data_power=None)
    delta_chi2_profile.append(-2 * ll - (-2 * chi2_baseline))
    print(f"  epsilon = {eps:.2f}  Δχ² = {delta_chi2_profile[-1]:+.3f}")

delta_chi2_profile = np.array(delta_chi2_profile)

# Plot the profile
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epsilon_grid, delta_chi2_profile, 'o-', color='teal', linewidth=2)
ax.axhline(0,  color='k',    linestyle='--', linewidth=0.8, label='ΛCDM baseline')
ax.axhline(1,  color='gray', linestyle=':',  linewidth=0.8, label='1σ threshold')
ax.axhline(4,  color='gray', linestyle='-.',  linewidth=0.8, label='2σ threshold')
ax.axvline(1,  color='coral',linestyle='--', linewidth=1.0, label='Full DDE (Union3)')
ax.set_xlabel('Template amplitude ε  (0=ΛCDM, 1=full DDE)', fontsize=12)
ax.set_ylabel(r'$\Delta\chi^2$  (DDE − baseline)', fontsize=12)
ax.set_title('Profile likelihood: DDE spectral tilt amplitude', fontsize=12)
ax.legend(fontsize=9)
ax.invert_yaxis()   # convention: deeper = better fit
plt.tight_layout()
plt.savefig('dde_profile_likelihood.png', dpi=150)
print("\nProfile plot saved to dde_profile_likelihood.png")
