# 	Author: 	Matthew Ogden
This cluster directory is my attempt to get SPAM working on our local cluster.

1) 	Execute.sh  ( on local minion )
	- syncs this directory to babbage
	- ssh into babbage and run prep.sh to begin execution on babbage.

2) 	prep.sh  ( Executes on Babbage head node )
	- Compiles MPI c code 
	- Submits qsub file to cluster work flow thingie....

3)  QSUB thingie  ( Wherever qsub thingie is )
	- Give important info to cluster workflow thingie 
	  - Name of job
	  - Number of cores I want
	  - What machine types to I want to run on 
	- Begins execution of SPAM_mpi.c among nodes

4)  cCluster.c 	( begins execution on all individual nodes requested )
	- Initializes MPI Environment 
	- Gets number of processors and my rank among them
	- Exectue Python script and pass my rank and comm size

5)  pyCluster.py   ( all nodes )
	- This script exists primarily because I know to get things done quickly via python
	- Now that all initialization of nodes has begun, this script gets what I want done
	  - Reads my rank among the comm size
	  - Copy SPAM source code to a unique directory
	  - Compile the SPAM code, (just in case nodes have different architecture?  I suspect his was redundant)
	  - Read giant list of runs to accomplish. 
	  - Call zooRun.py for each run with proper arguments

6) 	zooRun.py  ( all nodes )
	- This is my primary scirpt for running the basic_run spam code
	- has many command line options
	- runs basic_run and moves particle files where desired

7) 	basic_run   ( all nodes )
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
	  Files are still being copied to SPAM_data so it's still doing runs

  2) I let it run.  I checked the prog.txt files (Files my program would write to during execution)
	  I discovered that some nodes were operating much different speeds than others.
	  Indicating that I will likely want to do a master tasker and slave nodes in the future.
	  And my calculation of how long it'll take is likely way off...  Way off...
	  In the first 24 hours, most nodes are at 30 runs out of 580, some at 13, few at 90. 
	  30 runs in 24 hours... Little less than 1 an hr, it'll be a long time before it acheives 580.
	  This may explain why the cluster seemed much slower than my minion. 
	  I'm probably going to request another couple hundred clusters next week. 
	
  3) My deleting of run files on local nodes does not seem to be working...  
	  But new temp files should be overwriting old files since they're the same name....
	  May need to double check in case nodes are getting bogged down from too much diskspace.
	  
Lessons Learned
1) 	In a cluster, don't assume the nodes all run at the same speed as your local minion.
2)  For the cluster, take note of your job number.  It may disappear in qstat...
3)  Develop master/slave system among nodes.  This will account for variable speed between nodes/runs.
4)  Consider finding the fastest nodes on cluster for my own reasons...
5)  Consider making 2nd thread to retrieve next job while working on current job (Extra work)




Thoughts to self for master/slave implementation.

1) 	Use MPI c code for communication...   Would require me to take most of my tasks in the pyCluster script and move to c and learn how to accomplish master/slave communication when communication is uneven accross nodes. This is the most professional and common approach but may take the longest to implement... And  Ihave a bias against learning c atm... I know, very unprofessional of me. 

2) Can the python scripts take care of master/worker arrangement?... Can python scripts.. establish communications between each other?  I feel like it has something to do with sockets... Unknown methods and time needed to learn.  pass for now.

3)  Use a shared file on localstorage to communicate. Each node reads, takes one line/run, and rewrite the file without the line.  
	Will need robust collision avoidance.  
	Use another file as key?  
	Consider giving all nodes one run off the bat to prevent the initial collision for file access. 
	Seems easiest and quickest to implement, but will liekly lead to unexpected complications.  
	How would I detect if another process has a file open? 

4) Master python script could read main list and create smaller list of runs and send to each nodes directory.  
	The nodes would read their local list containing 2-5 runs queued. 
	This reduces collisions between 100+ nodes to one file, to 100 1-1 collision opportunities.
	My intuition tells me there's less of a chance between collision detection if it's 1 on 1, even if there's a 100 going on at the same time.  
	The master could do a linux command to get the number of runs in the queue.  Therefore never opening it just to check.
	If master is on 1-5 second loop, then if a line is missing, it means the worker just finished and is working on another run and won't be accessing it for another min (assuming 100k+ particles).   
	Maybe!   Maybe only place one run in file, worker will read file and delete it...

5)  Master's sole job is creating a single file with a run in seperate node directories.
	Once worker has read file, it deletes file and starts run. 
	If file doesn't exist, Master knows to create another one.
	Seems simple...
	Prevents file collision...

	Possible problems...?
	master messes up and all nodes are in limbo...
	
	Trouble shoot
	if it's been over 30 seconds and no file appears, exit program assuming completion.
	If file has phrase 'Completed', close down as well.

	Okay.  Let's work on this over a couple hours  and find out.
