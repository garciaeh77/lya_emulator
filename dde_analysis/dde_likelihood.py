"""
dde_likelihood.py

Subclass of LikelihoodClass that injects the Union3 DDE spectral
tilt template from Garza et al. 2026 (arXiv:2601.00767) into the
emulator prediction before the likelihood is evaluated.

The injection happens on the native Mpc/h k-grid, before rebinning
to km/s units, as recommended by Simeon Bird.
"""

import numpy as np
from lyaemu.likelihood import LikelihoodClass
from lyaemu import flux_power
from .template import make_template_interpolator


class DDELikelihood(LikelihoodClass):
    """
    Identical to LikelihoodClass except get_predicted multiplies
    the emulator flux power by the Union3 DDE template T(k, z, epsilon)
    before rebinning to km/s units.

    Parameters
    ----------
    epsilon : float
        Template amplitude. 0 = LCDM, 1 = full Union3 DDE prediction.
    All other arguments passed directly to LikelihoodClass.
    """

    def __init__(self, *args, epsilon=1.0, **kwargs):
        self.epsilon = epsilon
        self.template_fn = make_template_interpolator()
        super().__init__(*args, **kwargs)

    def get_predicted(self, params):
        """
        Override get_predicted to inject the DDE template on the
        native Mpc/h k-grid before rebinning to km/s.
        """
        # ── Step 1: replicate parent mean flux logic ──────────────────────
        nparams = params
        if self.mf_slope:
            from lyaemu import mean_flux as mflux
            tau0_fac = mflux.mean_flux_slope_to_factor(
                self.zout, params[0]
            )
            nparams = params[1:]
        else:
            tau0_fac = None

        # ── Step 2: get raw GP prediction in Mpc/h units ─────────────────
        predicted_nat, std_nat = self.gpemu.predict(
            np.array(nparams).reshape(1, -1),
            tau0_factors=tau0_fac
        )

        # ── Step 3: get h and omega_m from parameter vector ───────────────
        ndense = len(self.emulator.mf.dense_param_names)
        hindex = ndense + self.emulator.param_names["hub"]
        omindex = ndense + self.emulator.param_names["omegamh2"]
        h = nparams[hindex]
        omega_m = nparams[omindex] / h**2

        # ── Step 4: inject DDE template on native Mpc/h k-grid ───────────
        n_z = len(self.zout)
        n_k = len(self.gpemu.kf)

        pred = predicted_nat[0].reshape(n_z, n_k).copy()
        std  = std_nat[0].reshape(n_z, n_k).copy()

        for i, z in enumerate(self.zout):
            # Convert k from h/Mpc to s/km:
            # k [s/km] = k [h/Mpc] * h / (H(z)/(1+z) [km/s/Mpc])
            Hz_over_H0 = np.sqrt(omega_m * (1+z)**3 + (1 - omega_m))
            Hz_kms_mpc = h * 100.0 * Hz_over_H0
            k_kms = self.gpemu.kf * h / (Hz_kms_mpc / (1+z))

            T = self.template_fn(k_kms, z, epsilon=self.epsilon)
            pred[i] *= T
            std[i]  *= T

        # ── Step 5: rebin to km/s units (same as parent) ─────────────────
        okf, predicted_kms = flux_power.rebin_power_to_kms(
            kfkms=self.kf,
            kfmpc=self.gpemu.kf,
            flux_powers=pred.reshape(1, -1)[0],
            zbins=self.zout,
            omega_m=omega_m
        )
        _, std_kms = flux_power.rebin_power_to_kms(
            kfkms=self.kf,
            kfmpc=self.gpemu.kf,
            flux_powers=std.reshape(1, -1)[0],
            zbins=self.zout,
            omega_m=omega_m
        )

        return okf, predicted_kms, std_kms
