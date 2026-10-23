#!/bin/bash
# E6a: 5-fold cross-validation of the DL-S2 model, folds grouped by run.
# Run from galStuff/tng-exp-oct-2026 (Git Bash or WSL):  bash scripts/run_cv.sh [name] [epochs] [image]
# Each fold trains a fresh model with one fold of runs held out for validation.
# Output: results/<name>-f<k>.txt, results/<name>-f<k>-best-pred.npz, models/<name>-f<k>-best.h5, logs/<name>-f<k>.log
# Note: runName must not contain "test" (that switches the script into notebook mode).

NAME=${1:-cv5}
EPOCHS=${2:-20}
IMAGE=${3:-e6a-tf291-gpu}
GPU=${GPU:---gpus all}   # set GPU="" to run on CPU
HERE=$(pwd -W 2>/dev/null || pwd)

mkdir -p logs
for k in 0 1 2 3 4; do
    echo "=== fold $k  $(date)"
    MSYS_NO_PATHCONV=1 docker run --rm $GPU -v "$HERE:/work" -w /work "$IMAGE" \
        python scripts/train-fitness-score.py \
        -runName "$NAME-f$k" -data_loc data/dl-s2-10 \
        -split run -folds_file data/dl-s2-10-folds.json -fold $k -seed 2026 \
        -start 0 -stop "$EPOCHS" -num_epochs 1 -batch_size 32 -verbose 2 \
        > "logs/$NAME-f$k.log" 2>&1
    echo "    exit $?"
    grep -E "Best:" "results/$NAME-f$k.txt" | tail -1
done
echo "=== done $(date).  Summarise with:  python scripts/summarize_cv.py $NAME"
