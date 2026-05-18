#!/bin/bash
#SBATCH --job-name=ddeeps_largemem
#SBATCH --output=/home1/hgarciae/lya_emulator/dde_analysis/results/chain_logs/ddeeps_largemem_%j.out
#SBATCH --error=/home1/hgarciae/lya_emulator/dde_analysis/results/chain_logs/ddeeps_largemem_%j.err
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
mkdir -p "$REPO/dde_analysis/results/chains_dde_epsilon"
mkdir -p "$REPO/dde_analysis/results/covmats"

export USE_DDE_COVMAT="${USE_DDE_COVMAT:-0}"
export COVMAT_DDE="${COVMAT_DDE:-$REPO/dde_analysis/results/covmats/simeon_plus_epsilon_15p.covmat}"

if [ "$USE_DDE_COVMAT" = "1" ] && [ ! -f "$COVMAT_DDE" ]; then
  echo "ERROR: USE_DDE_COVMAT=1 but missing COVMAT_DDE at $COVMAT_DDE"
  exit 2
fi

echo "=== Preflight import check ==="
python -c "import configobj, torch, cobaya, mpi4py, h5py; from fake_spectra import spectra; from lyaemu.likelihood import LikelihoodClass; print('preflight ok')"
echo "=== Starting sampled-epsilon chain ==="
echo "REPO=$REPO"
echo "USE_DDE_COVMAT=$USE_DDE_COVMAT"
if [ "$USE_DDE_COVMAT" = "1" ]; then
  echo "COVMAT_DDE=$COVMAT_DDE"
else
  echo "COVMAT_DDE=disabled"
fi

mpirun -np 4 python dde_analysis/run_cobaya_epsilon_mcmc.py
