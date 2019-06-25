# 	Author: 	Matthew Ogden

This cluster directory is my attempt to get SPAM working on our local cluster.

1) 	Execute.sh  ( on local machine )
	- syncs my this directory to babbage
	- ssh into babbage and run prep.sh to begin execution on babbage.

2) 	prep.sh  ( Executes on Babbage head node )
	- Compiles MPI c code 
	- Submits qsub file to cluster work flow thingie....

3)  QSUB thingie  ( Wherever qsub thingie goes )
	- Give import info to cluster workflow thingie 
	  - Name of job
	  - Number of cores I want
	  - What machine types to I want to run on 

	- Begins execution of SPAM_mpi.c

4)  SPAM_mpi.c 	( begins execution on all individual nodes requested )
	- Initializes MPI Environment 
	- Gets number of processors and my rank among them
	- Exectue Python script and pass my rank and comm size

5)  clusterRun.py   ( on all individual nodes )
	- This script exists primarily because I know to get things done quickly via python
	- Now that all initialization of nodes has begun, this script gets what I want done
	  - Reads my rank among the comm size
	  - Copy SPAM source code to a unique directory
	  - Compile the SPAM code
	  - Run SPAM code....  ( need to remember how SPAM code works, be back soon)
