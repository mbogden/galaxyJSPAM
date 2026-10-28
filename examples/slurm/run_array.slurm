#!/bin/bash
# One array task per line of a commands file: task N runs line N on its own whole node(s),
# exactly as run_node.slurm would. Blank lines and lines starting with # are skipped and
# don't count.
#
# Submit with submit_array.sh, which sizes the array from the file:
#        submit_array.sh <commands-file> [max-at-once] [extra sbatch options...]
# By hand: sbatch --array=1-<lines>%<max-at-once> run_array.slurm /full/path/to/commands.txt
#
# Each line runs as `bash -c "<line>"` inside the image, once per rank.
#SBATCH --job-name=gtc-array
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/galStuff/logs/%x-%A_%a.log

export REPO=${REPO:-$HOME/galStuff/galactic_tidal_constraints}
COMMANDS=$1

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ ! -f "$COMMANDS" ]; then
    echo "Usage: sbatch --array=1-N run_array.slurm /full/path/to/commands.txt (or use submit_array.sh)" >&2
    exit 2
fi

line=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$COMMANDS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$line" ]; then
    echo "No line $SLURM_ARRAY_TASK_ID in $COMMANDS" >&2
    exit 2
fi

echo "commands:  $COMMANDS, line $SLURM_ARRAY_TASK_ID of array $SLURM_ARRAY_JOB_ID"
exec bash "$REPO/examples/slurm/run_node.slurm" bash -c "$line"
