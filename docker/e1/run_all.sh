#!/bin/bash
# Run every single-node E1 check inside the container: GPU/PyTorch/Keras, Fortran, MPI with 2 ranks.
# Usage: bash /opt/e1/run_all.sh
# Multi-node MPI (check 5) is launched from the host with srun; see README.md.
status=0
echo "=== container: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2), host $(hostname)"

python /opt/e1/env_check.py || status=1

echo; echo "=== 6. Fortran"
bash /opt/e1/fortran_check.sh || status=1

echo; echo "=== 4. MPI, 2 ranks"
# Docker runs as root; OpenMPI refuses that unless told otherwise.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -n 2 --oversubscribe python /opt/e1/mpi_check.py || status=1

echo; [ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
