# This script is for convenience to compile mpi job and begin server

mpicc -o spam_mpi_test cluster/cCluster.c -lm
qsub cluster/SPAM_testing.qsub
