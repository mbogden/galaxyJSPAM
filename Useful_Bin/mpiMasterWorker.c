/* 	This is a  simple program for me to learn how to use babbage
 *  	Has transited to initiating python script for SPAM runs
 * */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"


int main( int argc, char *argv[] ){
  int nProcs;
  int myRank;
  int nameLen;
  char totStr[256];

  char destStr[1000];
  int strLen = 1000;

  int strTag = 1;
  int statusTag = 2;

  const char * fileLoc;
  fileLoc = argv[1];

  char procName[MPI_MAX_PROCESSOR_NAME];

  // Get MPI Comm world data
  MPI_Init(&argc, &argv);
  MPI_Comm_size( MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myRank);
  MPI_Get_processor_name( procName, &nameLen);

  printf( "printf: Process %d on %d -- %s\n", myRank, nProcs, procName);

  if ( myRank == 0 ) {

	printf( "Reading file %s\n", fileLoc);
	FILE *fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;

	fp = fopen( fileLoc, "r");

	int toRank;
	int waitFlag;
	int collect;

	MPI_Status status;

	while (( read = getline( &line, &len, fp)) != -1) {

	  // Wait until master finds a worker ready to work
	  waitFlag = 0;

	  // Wait until a worker says they're ready
	  MPI_Recv( &toRank, 1, MPI_INT, MPI_ANY_SOURCE, statusTag, MPI_COMM_WORLD, &status);

	  // Send job to worker
	  printf("0 sending to %d\n",toRank);
	  MPI_Send( line, strlen( line)+1, MPI_CHAR, toRank, strTag, MPI_COMM_WORLD);

	}

	fclose(fp);


	line = "end";
	for ( int i=1; i<nProcs; i++){
	  MPI_Send( line, strlen( line)+1, MPI_CHAR, i, strTag, MPI_COMM_WORLD);
	}

  }

  // Else I'm a worker
  else {
	printf("%d Started\n",myRank);

	int keepWorking= 1;
	MPI_Request request = MPI_REQUEST_NULL;

	// While master is still sending jobs
	while( keepWorking == 1){

	  // Send message to master indicating worker ready to work
	  MPI_Isend( &myRank, 1, MPI_INT, 0, statusTag, MPI_COMM_WORLD, &request );

	  // Wait until message is received containing line executable
	  MPI_Recv( &destStr, strLen, MPI_CHAR, 0, strTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  printf("%d received line %s\n",myRank,destStr);

	  // Check if worker is done
	  if ( strcmp( "end", destStr) == 0 ){
		keepWorking = 0;
		continue;
	  }

	  // add uID to line
	  // execute

	  sleep(3);

	}

	printf("%d done working\n",myRank);

  }

	


  // Create string of python command.  with comm size and rank
  //sprintf(totStr,"python ~/cluster/pyCluster.py -cr %d -cs %d",myId,nProcs);

  // Run command
  //system(totStr);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

}

