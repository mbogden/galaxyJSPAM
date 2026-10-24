"""E1 checks 4-5: MPI ranks talk to each other, each rank on its own GPU.

Run with 2+ ranks, e.g. `mpirun -n 2 python mpi_check.py` or `srun --mpi=pmix -n 2 ...`.
Rank 0 prints PASS/FAIL; exits non-zero on failure.
"""
import os
import socket
import sys

import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

if torch.cuda.is_available():
    device = torch.device("cuda", local_rank % torch.cuda.device_count())
    gpu = torch.cuda.get_device_name(device)
else:
    device, gpu = torch.device("cpu"), "none"

value = (torch.ones(1, device=device) * (rank + 1)).item()
total = comm.allreduce(value, op=MPI.SUM)
rows = comm.gather((rank, socket.gethostname(), str(device), gpu), root=0)

if rank == 0:
    print(f"MPI: {MPI.Get_library_version().splitlines()[0]}")
    for r, host, dev, name in rows:
        print(f"rank {r}: {host}, {dev}, {name}")
    expected = size * (size + 1) / 2
    ok = size >= 2 and total == expected
    print(f"allreduce sum = {total} (expected {expected}), ranks = {size}")
    print(f"{'PASS' if ok else 'FAIL'} mpi")
    sys.exit(0 if ok else 1)
