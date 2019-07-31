# This script is for convenience to compile mpi job and begin server
#mpicc -o mpi_test galaxyJSPAM/Useful_Bin/cluster/mpiMasterWorker.c -lm
qsub sync_folder/Param_Finder/qParamFinder.qsub
