/* 	This is a  simple program for me to learn how to use babbage
 *  	Has transited to initiating python script for SPAM runs
 * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

int main( int argc, char *argv[]){
  int nProcs;
  int myId;
  int nameLen;
  char totStr[128];

  char procName[MPI_MAX_PROCESSOR_NAME];

  // Get MPI Comm world data
  MPI_Init(&argc, &argv);
  MPI_Comm_size( MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myId);
  MPI_Get_processor_name( procName, &nameLen);

  printf( "printf: Process %d on %d -- %s\n", myId, nProcs, procName);

  // Create string of python command.  with comm size and rank
  sprintf(totStr,"python ~/cluster/pyCluster.py -cr %d -cs %d",myId,nProcs);

  // Run command
  system(totStr);
  

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

}

