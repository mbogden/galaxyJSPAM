#!/bin/bash
# E6a: 5-fold cross-validation of the DL-S2 model on Babbage, one fold per array task.
# Submit from ~/galStuff/tng-exp-oct-2026 (logs/ must exist first):
#     mkdir -p logs
#     sbatch scripts/cv_babbage.sbatch                          # defaults: cv5, 20 epochs
#     sbatch --export=ALL,NAME=cv5-e40,EPOCHS=40 scripts/cv_babbage.sbatch
#     sbatch --array=2 scripts/cv_babbage.sbatch                # rerun one fold
# Then summarise:  apptainer exec e6a-tf291-gpu_v1.sif python scripts/summarize_cv.py <NAME>
# Note: NAME must not contain "test" (that switches the script into notebook mode).
#SBATCH --job-name=e6a-cv
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A-f%a.log
##SBATCH --partition=<name>     # set if Babbage needs an explicit GPU partition

NAME=${NAME:-cv5}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-32}
SIF=${SIF:-e6a-tf291-gpu_v1.sif}
k=$SLURM_ARRAY_TASK_ID

cd "$SLURM_SUBMIT_DIR"
echo "fold $k of $NAME on $(hostname), GPU $CUDA_VISIBLE_DEVICES, $(date)"

apptainer exec --nv --bind "$PWD:/work" --pwd /work "$SIF" \
    python scripts/train-fitness-score.py \
    -runName "$NAME-f$k" -data_loc data/dl-s2-10 \
    -split run -folds_file data/dl-s2-10-folds.json -fold "$k" -seed 2026 \
    -start 0 -stop "$EPOCHS" -num_epochs 1 -batch_size "$BATCH" -verbose 2

echo "exit $? $(date)"
grep -E "Best:" "results/$NAME-f$k.txt" | tail -1
