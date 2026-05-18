#!/bin/bash
#SBATCH --job-name=lcdm_largemem
#SBATCH --output=/home1/hgarciae/lya_emulator/dde_analysis/results/chain_logs/lcdm_largemem_%j.out
#SBATCH --error=/home1/hgarciae/lya_emulator/dde_analysis/results/chain_logs/lcdm_largemem_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=largemem
#SBATCH --time=7-00:00:00

set -euo pipefail

# Allow overriding repo path if needed.
REPO="${REPO:-/home1/hgarciae/lya_emulator}"
VENV="${VENV:-$HOME/venvs/selfinus}"

source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$REPO/dde_analysis/results/chain_logs"
mkdir -p "$REPO/dde_analysis/results/chains_lcdm_epsilon0"
mkdir -p "$REPO/dde_analysis/results/covmats"

export USE_SIMEON_COVMAT="${USE_SIMEON_COVMAT:-0}"
export COVMAT_LCDM="${COVMAT_LCDM:-$REPO/dde_analysis/results/covmats/simeon_lcdm_14p.covmat}"
export COBAYA_RMINUS1_STOP="${COBAYA_RMINUS1_STOP:-0.01}"
export COBAYA_BURN_IN="${COBAYA_BURN_IN:-1000}"
export COBAYA_MAX_SAMPLES="${COBAYA_MAX_SAMPLES:-120000}"
export COBAYA_OUTPUT_SUFFIX="${COBAYA_OUTPUT_SUFFIX:-disc_largemem_r001}"

if [ "$USE_SIMEON_COVMAT" = "1" ] && [ ! -f "$COVMAT_LCDM" ]; then
  echo "ERROR: USE_SIMEON_COVMAT=1 but missing COVMAT_LCDM at $COVMAT_LCDM"
  exit 2
fi

echo "=== Preflight import check ==="
python -c "import configobj, torch, cobaya, mpi4py, h5py; from fake_spectra import spectra; from lyaemu.likelihood import LikelihoodClass; print('preflight ok')"
echo "=== Starting LCDM chain ==="
echo "REPO=$REPO"
echo "USE_SIMEON_COVMAT=$USE_SIMEON_COVMAT"
if [ "$USE_SIMEON_COVMAT" = "1" ]; then
  echo "COVMAT_LCDM=$COVMAT_LCDM"
else
  echo "COVMAT_LCDM=disabled"
fi
echo "COBAYA_RMINUS1_STOP=$COBAYA_RMINUS1_STOP"
echo "COBAYA_BURN_IN=$COBAYA_BURN_IN"
echo "COBAYA_MAX_SAMPLES=$COBAYA_MAX_SAMPLES"
echo "COBAYA_OUTPUT_SUFFIX=$COBAYA_OUTPUT_SUFFIX"

mpirun -np 4 python dde_analysis/run_cobaya_lcdm_epsilon0_mcmc.py
