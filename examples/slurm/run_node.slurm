#!/bin/bash
# Run one command on whole Babbage node(s): host mpirun starts one rank per core, each rank
# inside the Apptainer image. A command that doesn't use MPI runs once per rank, so use
# --ntasks-per-node=1 for those.
#
# Usage: sbatch run_node.slurm <command> [args...]
#        sbatch --nodes=2 run_node.slurm <command> [args...]           (64 ranks)
#        SIF=~/galStuff/images/<image>-<tag>.sif sbatch run_node.slurm ...
# Example: sbatch run_node.slurm python /opt/gtc/checks/mpi_check.py
#
# Why host mpirun and not srun: Babbage's Slurm has no pmix plugin and Ubuntu's OpenMPI has no
# PMI2 support (ENV-1, 2026-10-28). The image's OpenMPI (4.1.6) must match the host's.
#SBATCH --job-name=gtc-run
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/galStuff/logs/%x-%j.log

SIF=${SIF:-$HOME/galStuff/images/gtc-base-image-v1.sif}
REPO=${REPO:-$HOME/galStuff/galactic_tidal_constraints}
MODULES="apptainer gnu12/12.4.0 openmpi4/4.1.6"

if [ $# -eq 0 ]; then
    echo "Usage: sbatch run_node.slurm <command> [args...]" >&2
    exit 2
fi

# Batch jobs start with no modules loaded; openmpi4 is hidden until gnu12 is loaded.
module load $MODULES || exit 1

# Each rank runs in its own Apptainer user namespace, so OpenMPI's CMA shared-memory copy
# isn't available; turn it off instead of warning about it on every run.
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# The .sif's sha256 identifies the exact image; it is computed once and kept next to the file.
if [ ! -f "$SIF.sha256" ] || [ "$SIF" -nt "$SIF.sha256" ]; then
    sha256sum "$SIF" | cut -d' ' -f1 > "$SIF.sha256"
fi

task=${SLURM_ARRAY_TASK_ID:+ (array task $SLURM_ARRAY_TASK_ID)}
echo "=== gtc job $SLURM_JOB_ID$task · $(date '+%Y-%m-%d %H:%M:%S')"
echo "nodes:     $SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES × $SLURM_NTASKS_PER_NODE ranks = $SLURM_NTASKS)"
echo "partition: $SLURM_JOB_PARTITION"
echo "image:     $SIF"
echo "sha256:    $(cat "$SIF.sha256")"
echo "apptainer: $(apptainer --version | cut -d' ' -f3)    modules: $MODULES"
echo "git:       $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
echo "command:   $*"
echo "==="

start=$(date +%s)
mpirun apptainer exec "$SIF" "$@"
status=$?

echo "=== exit $status after $(( $(date +%s) - start )) s"
exit $status
