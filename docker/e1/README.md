# E1: base-image checks

Pass/fail checks for the project's container image: `nvidia/cuda:12.6.3-runtime-ubuntu24.04`
with Python 3.12, uv, OpenMPI 4.1.6, PyTorch 2.14.1 (`cu126`), Keras 3 (torch backend) and gfortran.

| # | Check | Script |
|---|---|---|
| 1 | GPU visible; GPU arch is in the PyTorch wheel; GPU matmul matches CPU | `env_check.py` |
| 2 | Small CNN trains in PyTorch on the GPU (time per epoch) | `env_check.py` |
| 3 | Same CNN in Keras, weights on the GPU | `env_check.py` |
| 4 | MPI allreduce, 2 ranks on 1 node, one GPU each | `mpi_check.py` |
| 5 | Same over 2 nodes (Babbage, if allocatable) | `mpi_check.py` |
| 6 | Tiny Fortran extension builds with f2py and imports | `fortran_check.sh` |

Each script prints `PASS`/`FAIL` and exits non-zero on failure. Every run logs the hostname,
and on Babbage the Apptainer version.

## Build
```bash
docker build -t e1-cuda:dev docker/e1
```

## Run
**Docker (Windows or WSL):**
```bash
docker run --rm --gpus all e1-cuda:dev bash /opt/e1/run_all.sh
```

**Apptainer on WSL:** plain `--nv` can't see the GPU on WSL; bind the WSL driver libraries.
```bash
apptainer build e1-cuda.sif docker-daemon://e1-cuda:dev
apptainer exec --nv --bind /usr/lib/wsl --env LD_LIBRARY_PATH=/usr/lib/wsl/lib e1-cuda.sif bash /opt/e1/run_all.sh
```

**Babbage:** pull from Docker Hub, then submit the job script.
```bash
apptainer pull ~/galStuff/images/e1-cuda.sif docker://ogdenm12/e1-cuda:<tag>
sbatch docker/e1/e1_babbage.slurm
```
For check 5 (2 nodes), resubmit with `--nodes=2 --ntasks-per-node=1 --gres=gpu:1`
and run only the `srun` line.
