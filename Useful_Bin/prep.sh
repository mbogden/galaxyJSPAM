# This script is for convenience to compile mpi job and begin server

mpicc -o mpi_test c_test/mpiMasterWorker.c -lm
qsub c_test/qc_test.qsub
