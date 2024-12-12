#!/usr/bin/env python
# coding: utf-8

"""
File: mpi_queue_manager.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2024-Nov-11

Description: This code abstacts my parallel queue processing. 

References:  
    Code is based on example: https://github.com/luca-s/mpi-master-slave/blob/master/examples/example1.py
    Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.
"""

# ================================ IMPORTS ================================ #
# Standard library imports
import logging
import os
import sys
import time

import numpy as np
import dill
import concurrent.futures
from mpi4py import MPI
from mpi_master_slave import Master, Slave, WorkQueue

# Add main project directory to import project modules
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu


# ================================ GLOBAL VARIABLES ================================ #
LOGGER = logging.getLogger()

# Define MPI tags
TAG_TASK = 1
TAG_ERROR = 2
TAG_RESULT = 3
TAG_TERMINATE = 4

# ================================ CLASSES ================================ 

class Queue_Manager():

    def __init__(self,in_function, hard_mode = None):

        # Save functions to run
        self.eval_function = in_function

        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.cpu_count = os.cpu_count()
        self.master = Master(range(1, self.size))
        
        # Determine mode:
        # If hard coded, use that
        if hard_mode == 'mpi':
            self._set_mpi_mode()
        elif hard_mode == 'local':
            self._set_local_mode()
        elif hard_mode == 'solo':
            self._set_solo_mode()

        # MPI mode if size > 1
        # Local futures mode if size < 2 and cpu_count > 1
        # Single-core mode if size < 2 and cpu_count == 1
        elif hard_mode is None:
            if self.size > 1:
                self._set_mpi_mode()
            elif self.size < 2 and self.cpu_count > 1:
                self._set_local_mode()
            else:
                self._set_solo_mode()
        else:
            LOGGER.error("Invalid Parallel Processing Mode: %s" % hard_mode)
            LOGGER.error("Valid Modes: 'mpi', 'local', 'solo'")
            raise ValueError("Invalid Parallel Processing Mode %s" % hard_mode)
    # end __init__

    def __del__(self):
        LOGGER.info(f"Queue Manager Deleted")
    # end __del__

    def terminate_workers(self):
        if self.rank == 0 and self.size > 1:
            self.master.terminate_slaves()
    # end terminate_workers

    def _set_mpi_mode(self,):
        self.mode = 'mpi'
        self.queue = WorkQueue(self.master)
        self.wait_time = 0.01
        LOGGER.info(f"MPI Manager initialized with {self.size} Workers in MPI mode.")
    # end _set_mpi_mode

    def _set_local_mode(self,):
        self.mode = 'local'
        self.local_workers = self.cpu_count
        LOGGER.info(f"Local Manager initialized with {self.local_workers} workers using local multiprocessing (futures).")
    # end _set_local_mode
    
    def _set_solo_mode(self,):
        self.mode = 'solo'
        LOGGER.info("Solo Loop Mode.")
    # end _set_solo_mode

    def set_parallel_mode(self, mode):
        if mode == 'mpi':
            self._set_mpi_mode()
        elif mode == 'local':
            self._set_local_mode()
        elif mode == 'solo':
            self._set_solo_mode()
        else:
            LOGGER.error("Invalid Parallel Processing Mode: %s" % mode)
            LOGGER.error("Valid Modes: 'mpi', 'local', 'solo'")
            raise ValueError("Invalid Parallel Processing Mode %s" % mode)

    def gen_simple_progress_string( self, n_tasks, n_completed, length=20, char='=' ):
        """
        Generate a simple progress string
        """
        n = int( n_completed / n_tasks * length )
        return f"[{char*n}{' '*(length-n)}] {n_completed}/{n_tasks}"
    # end gen_simple_progress_string

    def evaluate_tasks(self, task_list):
        """
        Determine which run method to call based on the mode.
        """

        # validate task_list
        if isinstance(task_list, list):
            n = len(task_list)
        elif isinstance(task_list, np.ndarray):
            n = task_list.shape[0]
        else:
            LOGGER.error("Invalid task_list type: %s" % type(task_list))
            LOGGER.error("Valid types: list, numpy.ndarray")
            raise ValueError("Invalid task_list type: %s" % type(task_list))
        
        # Verify there are tasks to run
        if n == 0:
            LOGGER.error("No tasks to evaluate")
            return []

        # Run tasks based on mode   
        if self.mode == 'mpi':
            return self._run_mpi_queue(task_list)
        elif self.mode == 'local':
            return self._run_local_processes(task_list)
        elif self.mode == 'solo':  # single_core
            return self._run_single_core(task_list)
        else:
            LOGGER.error("Invalid Parallel Processing Mode: %s" % self.mode)
            LOGGER.error("Valid Modes: 'mpi', 'local', 'solo'")
            raise ValueError("Invalid Parallel Processing Mode %s" % self.mode)
    # end evaluate_tasks

    def _run_single_core(self, task_list):
        """
        If running on a single core and size < 1 or size=1 with no extra cores,
        simply loop through the evaluation function.
        """
        n = len(task_list)
        compiled_results = {}
        curr_time = time.time()

        # Initial progress print
        prog_str = self.gen_simple_progress_string(n, 0)
        LOGGER.info("Solo Manager: Progress: %s" % (prog_str))

        # Loop through tasks
        for i, t in enumerate(task_list):
            try:
                compiled_results[i] = self.eval_function(t)
            except Exception as e:
                LOGGER.error(f"Single-core task failed: {e}")
                compiled_results[i] = None

            if time.time() - curr_time > 1:
                prog_str = self.gen_simple_progress_string(n, len(compiled_results))
                LOGGER.info("Solo Manager: Progress: %s" % (prog_str))
                curr_time = time.time()

        prog_str = self.gen_simple_progress_string(n, len(compiled_results))
        LOGGER.info("Solo Manager: Progress: %s - Queue Complete" % (prog_str))
        return [compiled_results[i] for i in range(n)]
    # end _run_single_core

    def _run_mpi_queue(self, task_list):
        n = len(task_list)
        curr_time = time.time()
        compiled_results = {}

        # Add tasks to MPI queue
        for i, task in enumerate(task_list):
            self.queue.add_work(data=(i, task))
        
        # Do initial progress print
        prog_str = self.gen_simple_progress_string(n, 0)
        LOGGER.info("MPI Manager: Progress: %s" % (prog_str))

        # Keep 'poking' workers while there is work to do
        while not self.queue.done():
            self.queue.do_work()

            completed_results = self.queue.get_completed_work()
            for result in completed_results:
                worker_done, worker_results = result
                if worker_done:
                    idx, return_value = worker_results
                    LOGGER.debug(f'Manager: Received results {idx}:{return_value}')
                    compiled_results[idx] = return_value

            if time.time() - curr_time > 1:
                prog_str = self.gen_simple_progress_string(n, len(compiled_results))
                LOGGER.info("MPI Manager: Progress: %s" % (prog_str))
                curr_time = time.time()

            time.sleep(self.wait_time)

        # Queue is empty
        prog_str = self.gen_simple_progress_string(n, len(compiled_results))
        LOGGER.info("MPI Manager: Progress: %s - Queue Complete" % (prog_str))
        return [compiled_results[i] for i in range(n)]
    # end _run_mpi_queue

    @staticmethod
    def run_task(serialized_func, serialized_args):
        # Deserialize the function and arguments
        func = dill.loads(serialized_func)
        args = dill.loads(serialized_args)
        try:
            return func(args)
        except Exception as e:
            LOGGER.error(f"Task failed: {e}")
            return None
    # end run_task

    def _run_local_processes(self, task_list):
        n = len(task_list)
        curr_time = time.time()
        compiled_results = {}

        # Initial progress print
        prog_str = self.gen_simple_progress_string(n, 0)
        LOGGER.info(f"Local Manager: Progress: {prog_str}")

        # Run tasks using local processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.local_workers) as executor:
            future_to_idx = {
                executor.submit(
                    Queue_Manager.run_task,
                    dill.dumps(self.eval_function),  # Serialize the function
                    dill.dumps(task,)  # Serialize the arguments
                ): i for i, task in enumerate(task_list)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                compiled_results[i] = future.result()

                if time.time() - curr_time > 1:
                    prog_str = self.gen_simple_progress_string(n, len(compiled_results))
                    LOGGER.info(f"Local Manager: Progress: {prog_str}")
                    curr_time = time.time()

        prog_str = self.gen_simple_progress_string(n, len(compiled_results))
        LOGGER.info(f"Local Manager: Progress: {prog_str} - Queue Complete")
        return [compiled_results[i] for i in range(n)]
    # end _run_local_processes

# end Queue_Manager

class Queue_Worker(Slave):
    """
    Worker extends Slave class, overrides the 'do_work' method
    and calls 'Slave.run'. The Master will do the rest
    """

    def __init__(self, in_function ):
        super(Queue_Worker, self).__init__()

        # Get MPI rank
        self.rank = MPI.COMM_WORLD.Get_rank()

        # Save dictionary of possible functions to run
        self.eval_function = in_function
        LOGGER.info(f"Queue Worker {self.rank}: Initialized")

        # Start worker
        self.run()

    def __del__(self):
        LOGGER.info(f"Queue Worker {self.rank}: Deleted")
    
    def do_work(self, in_message):

        task_idx, task_input = in_message

        LOGGER.debug( f"Worker {self.rank}: Received task {task_idx}" )

        try:
            # Choose functcion to run based on given task
            task_output = self.eval_function(task_input)
        except Exception as e:
            LOGGER.error( f"Worker {self.rank}: Task {task_idx} failed: \n{e}" )
            task_output = None

        LOGGER.debug( f"Worker completed task: {task_idx}" )
        return (True, (task_idx, task_output))
# end Queue_Worker

# ================================ FUNCTIONS ================================ #
# setup two functions functions to evaluate
def test_func_1(x, a=1, b=2):
    time.sleep( 0.25 )
    return np.sum( a * np.power(x, b) )

# Computationally expensive function
def test_func_2(x, a=1, b=2):
    # randomly wait between 0 and 10 seconds
    time.sleep( np.random.random() * 5 )
    return np.sum( a * np.power(x, b) ) * np.random.random()

# Wrapper around both functions for demonstration
def func_wrapper( w_input ):
    f_id, f_input = w_input
    if f_id == 1:
        return test_func_1(f_input)
    elif f_id == 2:
        return test_func_2(f_input)
# end func_wrapper

# ================================ TESTING ================================ #
def test_queue( ):
    global LOGGER

    # Example function calls
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Have only rank 0 print to avoid duplicate prints
    if rank == 0:
        print("\nTESTING QUEUE MANAGER\n")

    # Setup Workers, and have them on standby
    if rank != 0:
        worker = Queue_Worker( func_wrapper )
        del worker
    
    # Setup manager and run tests
    if rank == 0:
        try:
            # Solo mode
            time.sleep(1)  # Let workers initialize and print their messages
            print("\nTEST 1: SOLO MODE\n")
            start_time = time.time()
            manager = Queue_Manager( func_wrapper, hard_mode='solo' )
            n = 4
            test_list = [ (2, np.random.random(10)) for i in range(n) ]
            results = manager.evaluate_tasks( test_list )
            time_taken = time.time() - start_time
            time.sleep(1)

            print(f"\nTest 1 Results: {len(results)/time_taken:.2f} tasks per second")
            gu.tabprint(f"Results: {len(results)}")
            gu.tabprint(f"Time Taken: {time_taken:.2f} seconds")
            gu.tabprint(f"Num Cores: 1")
            time.sleep(1)
            
            # Local mode
            print("\n# ================================ #\n")
            print("\nTEST 2: LOCAL MACHINE MODE\n")
            manager.set_parallel_mode('local')
            n = 100
            test_list = [ (2, np.random.random(10)) for i in range(n) ]
            start_time = time.time()
            results = manager.evaluate_tasks( test_list )
            time_taken = time.time() - start_time
            time.sleep(1)

            print(f"\nTest 2 Results: {len(results)/time_taken:.2f} tasks per second")
            gu.tabprint(f"Results: {len(results)}")
            gu.tabprint(f"Time Taken: {time_taken:.2f} seconds")
            gu.tabprint(f"Num Cores: {manager.local_workers}")
            time.sleep(1)

            # MPI mode
            print("\n# ================================ #\n")
            print("\nTEST 3: MPI MODE\n")
            manager.set_parallel_mode('mpi')
            n = 20
            test_list = [ (2, np.random.random(10)) for i in range(n) ]
            start_time = time.time()
            results = manager.evaluate_tasks( test_list )
            time_taken = time.time() - start_time
            time.sleep(1)

            print(f"\nTest 3 Results: {len(results)/time_taken:.2f} tasks per second")
            gu.tabprint(f"Results: {len(results)}")
            gu.tabprint(f"Time Taken: {time_taken:.2f} seconds")
            gu.tabprint(f"Num Cores: {manager.size}")
            print('')

            # Test termination
            manager.terminate_workers()
            time.sleep(1)       
            del manager
        except Exception as e:
            LOGGER.critical("Error in test_queue: %s" % e)

# ================================ MAIN ================================ #
# If script is called main
if __name__ == '__main__':

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("\PARALLEL PROCESSING QUEUE MANAGER\n")
        gu.tabprint("This module is intended to be imported and used in other scripts.")
        gu.tabprint("Calling as main will run tests and examples.")
        print("\nInitializing Arguments, and Logger\n")

    try:
        args, LOGGER = gu.initialize_environment()
        # gu.change_logging_level('DEBUG')

    except:
        print("Failed to Initalize Arguments and Logger")
        sys.exit(1)

    # Run tests
    test_queue()

    comm.Barrier()
    if rank == 0:
        print("\nEnd of Script\n")
