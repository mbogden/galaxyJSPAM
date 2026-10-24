#!/bin/bash
# E1 on Babbage: all checks on one GPU node, then MPI across 2 ranks via Slurm (pmix) and host mpirun.
# Usage: sbatch e1_babbage.slurm            (SIF defaults to ~/galStuff/images/e1-cuda.sif)
#        SIF=/path/to/img.sif sbatch e1_babbage.slurm
#SBATCH --job-name=e1-base-image
#SBATCH --partition=research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=e1-%j.log

SIF=${SIF:-$HOME/galStuff/images/e1-cuda.sif}

echo "=== host: $(hostname) | $(apptainer --version) | job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo; echo "=== checks 1-3, 6 (one process)"
apptainer exec --nv "$SIF" python /opt/e1/env_check.py
apptainer exec --nv "$SIF" bash /opt/e1/fortran_check.sh

echo; echo "=== check 4a: srun --mpi=pmix, 2 ranks"
srun --mpi=pmix -n 2 apptainer exec --nv "$SIF" python /opt/e1/mpi_check.py

echo; echo "=== check 4b: host mpirun (OpenMPI 4.1.6), 2 ranks"
mpirun -n 2 apptainer exec --nv "$SIF" python /opt/e1/mpi_check.py
