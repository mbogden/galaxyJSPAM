# This script is for convenience to compile mpi job and begin server
cd galaxyJSPAM && git pull
cd ~
mpicc -o mpi_test galaxyJSPAM/Useful_Bin/cluster/mpiMasterWorker.c -lm
#qsub c_test/qc_test.qsub
