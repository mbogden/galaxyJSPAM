#!/bin/bash
# E6a: smoke test and timing on one Babbage GPU node (2 GPUs, 32 CPUs).
# Run inside an allocation (srun ... --pty bash), from ~/galStuff/tng-exp-oct-2026:
#     bash scripts/smoke_node.sh
# 1. Benchmark (no training) on GPU 0 with 1, 4, 8, 16 workers: loader throughput vs GPU step,
#    and a check that batches are identical whatever the number of workers.
# 2. Real training, 2 epochs of fold 0, two runs at once: 1 worker on GPU 0, 12 workers on GPU 1.
# Output: logs/smoke-*.log, results/smoke-*.txt, and a timing summary at the end.

SIF=${SIF:-e6a-tf291-gpu_v1.sif}
RUN="apptainer exec --nv --bind $PWD:/work --pwd /work $SIF python scripts/train-fitness-score.py \
     -data_loc data/dl-s2-10 -split run -folds_file data/dl-s2-10-folds.json -fold 0 -batch_size 32 -verbose 2"
mkdir -p logs results models
T0=$(date +%s)
echo "node $(hostname), $(nproc) CPUs, $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo; echo "=== 1. benchmark (GPU 0)"
for w in 1 4 8 16; do
    t=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 $RUN -runName smoke-bench-w$w -benchmark 40 -workers $w > logs/smoke-bench-w$w.log 2>&1
    echo "--- workers $w (exit $?, $(( $(date +%s) - t ))s incl. start-up)"
    sed -n '/^Benchmark:/,$p' logs/smoke-bench-w$w.log | grep -vE 'W tensorflow|I tensorflow'
done

echo; echo "=== 2. training, 2 epochs of fold 0: workers 1 on GPU 0 and workers 12 on GPU 1, in parallel"
t=$(date +%s)
CUDA_VISIBLE_DEVICES=0 $RUN -runName smoke-train-w1  -stop 2 -num_epochs 1 -workers 1  > logs/smoke-train-w1.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN -runName smoke-train-w12 -stop 2 -num_epochs 1 -workers 12 > logs/smoke-train-w12.log 2>&1 &
wait
echo "both done in $(( $(date +%s) - t ))s"
for r in w1 w12; do
    echo "--- $r"
    grep -E '^[0-9]+/[0-9]+ - [0-9]+s|^Timing|^Test Correlation' logs/smoke-train-$r.log
    grep -E 'Traceback|Error:' logs/smoke-train-$r.log | head -3
done

echo; echo "=== total $(( $(date +%s) - T0 ))s"
