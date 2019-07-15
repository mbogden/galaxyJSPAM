This directory 'Useful_Bin' is an experiment.  
It is meant to house code that has very specific purposes on how to run code.
Since I am currently running code on 3 different systems, I would like to simply how to how to run batches of code across systems and cores.
My goal is to make code that does something very simple.  

- cluster_mpi_master_worker.c
  # Speculation code.  Not actually created yet
  # This code is meant to run on our local babbage in an MPI entirement.  
  # It creates the MPI environment
  # It reads a file
  # It sends each line of the file to each worker.
  # Each worker executes that line as if it's an executable.
	# Ex. './basic_run.exe -arg1 -arg2'
	# Ex. 'python3 pixel_comparison.py
