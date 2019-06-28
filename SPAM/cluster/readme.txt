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

4)  cCluster.c 	( begins execution on all individual nodes requested )
	- Initializes MPI Environment 
	- Gets number of processors and my rank among them
	- Exectue Python script and pass my rank and comm size

5)  pyCluster.py   ( on all individual nodes )
	- This script exists primarily because I know to get things done quickly via python
	- Now that all initialization of nodes has begun, this script gets what I want done
	  - Reads my rank among the comm size
	  - Copy SPAM source code to a unique directory
	  - Compile the SPAM code
	  - Call zooRun.py with proper arguments

6) 	zooRun.py  ( on all individual nodes )
	- This is my primary scirpt for running the basic_run spam code
	- has many command line options
	- runs basic_run and moves particle files where desired

7) 	basic_run   ( on all individula nodes )
	- The actual program does does the work


In Prog notes.
I submitted the big job for 59,000 runs at 5:00 pm on Thur 27 JUN 2019.
I returned 4:00 pm the next day and discovered a few things.  

  1) My job was no longer showing when I used command 'qstat'. odd.. 
	  Presuming the job was canceled, I deleted the temporary files.  Only, they kept coming back.
	  My job was still running on the cores, but wasn't listed on the qstat for some reason.
	  Now I have no way of deleting it myself... And the system admin is not here for me to ask.
	  Also...  I may have missed up progress by deleting some of the files while it was running.
	  I put in a check before moving, but I didn't thoroughly test it. 

  2) I let it run.  I checked the prog.txt files (Files my program would write to during execution)
	  I discovered that some nodes were operating much different speeds than others.
	  Indicating that I will likely want to do a master tasker and slave nodes in the future.
	  And my calculation of how long it'll take is likely way off...  Way off...
	  In the first 24 hours, most nodes are at 30 runs out of 580, some at 13, few at 90. 
	  This may explain why the cluster seemed much slower than my minion. 
	  
Lessons Learned
1) 	In a cluster, don't assume the nodes all run at the same speed as your local minion.
2)  For the cluster, take note of your job number.  It may disappear in qstat...
3)  Develop master/slave system among nodes.  This will account for variable speed between nodes/runs.
4)  Consider finding the fastest nodes on cluster for my own reasons...
5)  Consider making 2nd thread to retrieve next job while working on current job


