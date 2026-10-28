#!/bin/bash
# Submit a commands file as one job array: one task per line, each on its own whole node(s).
# Usage: submit_array.sh <commands-file> [max-at-once] [extra sbatch options...]
#        submit_array.sh tng1_targets.txt              (at most 5 tasks at once)
#        submit_array.sh tng1_targets.txt 10 -t 12:00:00
# Blank lines and lines starting with # don't count. Cancel all tasks with `scancel <job id>`.
set -euo pipefail

if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "Usage: submit_array.sh <commands-file> [max-at-once] [extra sbatch options...]" >&2
    exit 2
fi
commands=$(realpath "$1")
max=${2:-5}
shift $(( $# >= 2 ? 2 : 1 ))

n=$(grep -c -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$commands" || true)
if [ "$n" -eq 0 ]; then
    echo "No commands in $commands" >&2
    exit 2
fi

here=$(dirname "$(realpath "$0")")
echo "Submitting $n tasks from $commands, at most $max at once"
sbatch --array=1-"$n"%"$max" "$@" "$here/run_array.slurm" "$commands"
