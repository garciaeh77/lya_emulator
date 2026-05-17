"""
Cobaya likelihood wrapper for full MCMC sampling of DDE template amplitude.

This class extends DDELikelihood so epsilon can be sampled jointly with the
standard emulator + nuisance parameters.
"""

import numpy as np
from cobaya.likelihood import Likelihood

from dde_analysis.dde_likelihood import DDELikelihood


class CobayaDDELikelihoodClass(Likelihood, DDELikelihood):
    """Cobaya-compatible likelihood class with sampled epsilon."""

    # Standard emulator/likelihood options (same pattern as lyaemu.likelihood)
    basedir: str
    HRbasedir: str = None
    mean_flux: str = "s"
    max_z: float = 4.6
    min_z: float = 2.2
    emulator_class: str = "standard"
    t0_training_value: float = 1.0
    optimise_GP: bool = True
    emulator_json_file: str = "emulator_params.json"
    data_corr: bool = True
    tau_thresh: int = None
    use_meant: bool = False
    sim_meant: float = None
    traindir: str = None
    data_power: float = None
    include_emu: bool = True
    loo_errors: bool = False
    hprior: bool = False
    oprior: bool = False
    bhprior: bool = False
    sdss: str = "dr14"

    # DDE amplitude (can be fixed in config or sampled as input param "epsilon")
    epsilon: float = 0.0

    # Required for Cobaya parameter parsing
    input_params_prefix: str = ""

    def initialize(self):
        DDELikelihood.__init__(
            self,
            self.basedir,
            HRbasedir=self.HRbasedir,
            mean_flux=self.mean_flux,
            max_z=self.max_z,
            min_z=self.min_z,
            emulator_class=self.emulator_class,
            t0_training_value=self.t0_training_value,
            optimise_GP=self.optimise_GP,
            emulator_json_file=self.emulator_json_file,
            data_corr=self.data_corr,
            tau_thresh=self.tau_thresh,
            use_meant=self.use_meant,
            traindir=self.traindir,
            loo_errors=self.loo_errors,
            sdss=self.sdss,
            epsilon=self.epsilon,
        )

    def logp(self, **params_values):
        # Sampled epsilon (if present) overrides fixed class epsilon.
        eps = float(params_values.get("epsilon", self.epsilon))
        self.epsilon = eps

        # Build parameter vector for emulator likelihood in Cobaya param order,
        # excluding epsilon because it is injected separately via self.epsilon.
        base_params = [p for p in self.input_params if p != "epsilon"]
        params = np.array([params_values[p] for p in base_params], dtype=float)

        return self.likelihood(
            params,
            include_emu=self.include_emu,
            data_power=self.data_power,
            use_meant=self.use_meant,
            hprior=self.hprior,
            oprior=self.oprior,
            bhprior=self.bhprior,
        )

