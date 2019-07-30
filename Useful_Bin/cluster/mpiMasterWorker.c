/* 	This is a  simple program for me to learn how to use babbage
 *  	Has transited to initiating python script for SPAM runs
 * */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

char *str_replace(char *orig, char *rep, char *with);
char *replaceWord(const char *s, const char *oldW, const char *newW);

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
	int collect;

	MPI_Status status;

	while (( read = getline( &line, &len, fp)) != -1) {

	  // Wait until a worker says they're ready
	  MPI_Recv( &toRank, 1, MPI_INT, MPI_ANY_SOURCE, statusTag, MPI_COMM_WORLD, &status);

	  // Send job to worker
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

	  printf("%d Sent master ready message\n",myRank);

	  // Wait until message is received containing line executable
	  MPI_Recv( &destStr, strLen, MPI_CHAR, 0, strTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  //
	  // Check if worker is done
	  if ( strcmp( "end", destStr) == 0 ){
		keepWorking = 0;
		continue;
	  }

	  // add uID to line
	  printf("%d before %s\n",myRank,destStr);
	  

	  char oldID[50] = "-uID";
	  char newID[50];

	  sprintf( newID,"-uID %d",myRank);
	  printf("replace %s\n",newID);
	  
	  char * cmd;
	  //cmd = replaceWord( destStr, oldID, newID);
	  cmd = str_replace( destStr, *oldID, *newID);
	  
	  printf("%d executing %s\n",*cmd);
	  //system(*cmd);

	}

	printf("%d done working\n",myRank);

  }


  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

}

// You must free the result if result is non-NULL.
char *str_replace(char *orig, char *rep, char *with) {
    char *result; // the return string
    char *ins;    // the next insert point
    char *tmp;    // varies
    int len_rep;  // length of rep (the string to remove)
    int len_with; // length of with (the string to replace rep with)
    int len_front; // distance between rep and end of last rep
    int count;    // number of replacements

    // sanity checks and initialization
    if (!orig || !rep)
        return NULL;
    len_rep = strlen(rep);
    if (len_rep == 0)
        return NULL; // empty rep causes infinite loop during count
    if (!with)
        with = "";
    len_with = strlen(with);

    // count the number of replacements needed
    ins = orig;
    for (count = 0; tmp = strstr(ins, rep); ++count) {
        ins = tmp + len_rep;
    }

    tmp = result = malloc(strlen(orig) + (len_with - len_rep) * count + 1);

    if (!result){
		printf("failed replace, returning\n");
        return NULL;
	}

    // first time through the loop, all the variable are set correctly
    // from here on,
    //    tmp points to the end of the result string
    //    ins points to the next occurrence of rep in orig
    //    orig points to the remainder of orig after "end of rep"
    while (count--) {
        ins = strstr(orig, rep);
        len_front = ins - orig;
        tmp = strncpy(tmp, orig, len_front) + len_front;
        tmp = strcpy(tmp, with) + len_with;
        orig += len_front + len_rep; // move to next "end of rep"
    }
    strcpy(tmp, orig);
    return result;
}



// Function to replace a string with another 
// string 
char *replaceWord(const char *s, const char *oldW, 
                                 const char *newW) 
{ 
    char *result; 
    int i, cnt = 0; 
    int newWlen = strlen(newW); 
    int oldWlen = strlen(oldW); 
  
    // Counting the number of times old word 
    // occur in the string 
    for (i = 0; s[i] != '\0'; i++) 
    { 
        if (strstr(&s[i], oldW) == &s[i]) 
        { 
            cnt++; 
  
            // Jumping to index after the old word. 
            i += oldWlen - 1; 
        } 
    } 
  
    // Making new string of enough length 
    result = (char *)malloc(i + cnt * (newWlen - oldWlen) + 1); 
  
    i = 0; 
    while (*s) 
    { 
        // compare the substring with the result 
        if (strstr(s, oldW) == s) 
        { 
            strcpy(&result[i], newW); 
            i += newWlen; 
            s += oldWlen; 
        } 
        else
            result[i++] = *s++; 
    } 
  
    result[i] = '\0'; 
    return result; 
} 
  
