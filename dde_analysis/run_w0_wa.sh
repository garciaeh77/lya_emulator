#!/bin/bash
#SBATCH --account=gluscevi_339
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=w0_wa
#SBATCH --output=/home1/hgarciae/yaml_files/output_logs/w0_wa_%j.log
#SBATCH --error=/home1/hgarciae/yaml_files/output_logs/w0_wa_%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load openmpi
module load python/3.10.16

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export COBAYA_PACKAGES=/home1/hgarciae/cobaya_packages

source ~/venvs/selfinus/bin/activate

cd /home1/hgarciae/yaml_files
mkdir -p output_logs

srun -n "${SLURM_NTASKS}" cobaya-run /home1/hgarciae/yaml_files/w0_wa.yaml --resume
