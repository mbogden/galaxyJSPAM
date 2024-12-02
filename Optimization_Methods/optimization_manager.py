#!/usr/bin/env python3
# coding: utf-8

"""
File: optimization_manager.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2024-Nov-11

References:  Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.
"""

# ================================ IMPORTS ================================ #
# Standard library imports
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from math import floor, ceil
from mpi4py import MPI
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.sampler import Lhs

# Add main project directory to import project modules
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu
import utilities.mpi_queue_manager as qm

# ================================ GLOBAL VARIABLES ================================ #
LOGGER = logging.getLogger()

# ================================ FUNCTIONS ================================ #

# ================================ CLASSES ================================ #
class Optimization_Manager():

    def __init__(self, black_box_function, space_info, save_loc=None ):

        LOGGER.debug('Optimization Manager: Initializing')

        if black_box_function is None:
            LOGGER.error('Optimization Manager: black_box_function is None')
            raise ValueError('Optimization Manager: black_box_function is None')
        
        self.black_box_function = black_box_function
        self.space_info = space_info

        # Gen space
        self.explored_points = None
        self.explored_values = None
        self.space = []
        for var_name, var_info in space_info.items():
            self.space.append(Real(var_info['bounds'][0], var_info['bounds'][1], prior=var_info.get('prior','uniform'), name=var_name, transform='normalize',))

        # Get mpi rank and size
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
        # Determine work role
        if self.rank == 0 and self.size == 1:
            self.work_role = 'solo'
            LOGGER.debug('Optimization Manager: Running Solo')

        elif self.rank == 0:
            self.work_role = 'master'
            LOGGER.debug('Optimization Manager: Running Master')

        else:
            self.work_role = 'worker'
            LOGGER.debug('Optimization Manager: Running Worker')
        
        # Open save_loc if given and not worker
        self.save_loc = None
        self.queue_manager = None

        if save_loc is not None:
            self.save_loc = save_loc
        
        if self.work_role == 'master':
            self.queue_manager = qm.Queue_Manager()
            # MPI.COMM_WORLD.barrier()

        elif self.work_role == 'worker':
            self.queue_manager = qm.Queue_Worker( {0:self.black_box_function} )
            # MPI.COMM_WORLD.barrier()

        LOGGER.info('Optimization Manager: Initialized at work role: %s' % self.work_role)

    # modify how class deletes itself
    def __del__(self):
        if self.work_role == 'master':
            LOGGER.debug('Optimization Manager: Terminating Workers')
            if self.queue_manager is not None:
                self.queue_manager.terminate_workers()
                del self.queue_manager
        LOGGER.debug('Optimization Manager: Deleted')
    
    def _generate_grid_points(self, space, num=10):
        grids = []
        for dim in space:
            if dim.prior == 'uniform':
                grids.append(np.linspace(dim.low, dim.high, num=num))
            elif dim.prior == 'log-uniform':
                grids.append(np.logspace(np.log10(dim.low), np.log10(dim.high), num=num))
            else:
                grids.append(np.linspace(dim.low, dim.high, num=num))  # Default to uniform
                
        grid_points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(space))
        return grid_points

    def _evaluate(self, inputs):

        new_pts = []
        new_val = []
        
        # Assuming only master or solo will call this function
        if self.work_role == 'solo':
            for x in inputs:
                y = self.black_box_function(x)
                new_pts.append( x )
                new_val.append( y )

        # If manager, use parallel processing queue
        elif self.work_role == 'master':

            # Wrap task to workers and get reults
            queue_list = [ (0, x) for x in inputs ]
            results = self.queue_manager.run(queue_list)

            # Add to explored points
            for i, x in enumerate( inputs ):
                new_pts.append( x )
                new_val.append( results[i] )                               
        
        new_pts = np.array(new_pts).reshape(-1, len(self.space))
        new_val = np.array(new_val)

        # Add new points to explored points
        if self.explored_points is None:
            self.explored_points = new_pts
            self.explored_values = new_val
        else:
            self.explored_points = np.concatenate( (self.explored_points, new_pts) )
            self.explored_values = np.concatenate( (self.explored_values, new_val) )

        return new_pts, new_val

    def save_pts(self):
        # save explored points as csv with headers from space_info
        pts_df = pd.DataFrame(self.explored_values, columns=['final_value'])

        # Loop through space_info and add columns to df
        for i, (var_name, var_info) in enumerate(self.space_info.items()):
            pts_df[var_name] = self.explored_points[:,i]

        LOGGER.debug('Points: ', self.explored_points.shape )
        print('Values: ', self.explored_values.shape )
        print('DF: ', pts_df.shape )

        if self.save_loc is not None:
            print("Saving to: ", self.save_loc)
            pts_df.to_csv(self.save_loc, index=False)
        else:
            print("Not")



    def initial_exploration(self, time_cutoff_sec=60, timing_points=4):
        
        # Assuming only master or solo will call this function
        if self.work_role == 'worker':
            LOGGER.error('Optimization Manager: Only master or solo can call initial_exploration')
            raise ValueError('Optimization Manager: Only master or solo can call initial_exploration')

        LOGGER.info('Optimization Manager: Estimating Time per Task for Exploration')
        start_time = time.time()
        num_dimensions = len(self.space)

        grid_points = self._generate_grid_points(self.space, num=5)

        # Do n timing points (per processer) to estimate time per execution
        if self.work_role == 'solo':
            selected_indices = np.random.choice(len(grid_points), size=timing_points, replace=False)  

        elif self.work_role == 'master':
            if timing_points * (self.size-1) > len( grid_points ):
                selected_indices = np.arange(len(grid_points))
            else:
                selected_indices = np.random.choice(len(grid_points), size=timing_points * (self.size-1), replace=False)
        
        # Package selected grid points for executions
        x_list = [ grid_points[i] for i in selected_indices ]
        self._evaluate(x_list)

        # Get sense of time per execution
        self.time_per_task = (time.time() - start_time) / (len(x_list) )
        LOGGER.info('Optimization Manager: Time per execution: %.2f s' % self.time_per_task)

        # Estimate number of tasks that can be done in time cutoff
        potential_num_tasks = time_cutoff_sec / self.time_per_task

        # Get rough idea of spacing between points needed for grid search
        num_points_per_dim = floor( potential_num_tasks ** (1 / num_dimensions) )

        # Create points to explore
        if num_points_per_dim >= 7:
            grid_points = self._generate_grid_points(self.space, num=num_points_per_dim)
            x_list = grid_points
            np.random.shuffle(x_list)
            LOGGER.info('Optimization Manager: Doing Grid Exploration: %d'% len(x_list))

        else:
            # Do LHS exploration
            lhs = Lhs(lhs_type="classic", criterion="maximin")
            x_list = lhs.generate(self.space, n_samples=int(potential_num_tasks))
            np.random.shuffle(x_list)
            LOGGER.info('Optimization Manager: Doing LHS exploration: %d'% len(x_list))

        # Explore points, but save results n seconds
        save_n_sec = 300
        n_pts_per_minute = ceil(save_n_sec / self.time_per_task)

        for i in range(1, ceil(len(x_list) / n_pts_per_minute) + 1):
            start = (i-1) * n_pts_per_minute
            end = min(i * n_pts_per_minute, len(x_list))
            new_pts, new_val = self._evaluate(x_list[start:end])

            # Save points every minute
            self.save_pts()

            # If time is up, break
            if time.time() - start_time > time_cutoff_sec:
                break
            
        return new_pts, new_val

    def pick_init_points(self, n_points):

        n_explored = len(self.explored_points)
        init_pts = []

        if n_explored < n_points:
            return self.explored_points

        # Get half as the most recent points
        init_pts = self.explored_points[int(-n_points/2):]

        # Randomly pick the other half
        random_indices = np.random.choice(n_explored, size=int(n_points/2), replace=False)
        init_pts.extend( self.explored_points[random_indices] )

        # prepare points and values
        initial_points = [ k for k in init_pts ]
        initial_values = [ v for v in init_pts.values() ]

        return (initial_points, initial_values)

    def setup_optimizer( self, time_cutoff_sec=10, timing_calls=4):
        
        """
        Start the optimization process using Gaussian Process Minimization.
        
        Parameters:
        - n_calls: Total number of calls to the objective function.
        - n_random_starts: Number of random starts before using the surrogate model.
        """

        LOGGER.info('Optimization Manager: Setting up Optimization')

        # If n_calls is not set, estimate number of calls based on time_running and self.time_per_task
        if n_calls is None:
            n_calls = floor(time_running / self.time_per_task)
            LOGGER.info('Optimization Manager: Estimated n_calls: %d' % n_calls)

        # Use built in function to get initial points for function
        if self.explored_points == {}:
            self.initial_exploration(time_running/10)

        # Collect initial points and their corresponding function values
        initial_points = list(self.explored_points.keys())
        initial_values = list(self.explored_points.values())

        # Get idea for how long optimization will take        
        self.pts_per_prediction = 100

        # Get some points to start with
        initial_points, initial_values = self.pick_init_points( self.pts_per_prediction )

        start_time = time.time()

        result = gp_minimize(
            func=self.black_box_function,
            dimensions=self.space,
            n_calls=4,
            n_random_starts=0,
            x0=initial_points,
            y0=initial_values,
            verbose=False
        )

        # Get time for prediction, by taking prediction time + time per task and subtracting time per task
        self.time_per_prediction = (time.time() - start_time) / timing_calls - self.time_per_task*timing_calls

        # Keep time to prediction under 10% of time per task
        while time_per_prediction < self.time_per_task * 0.1:
            self.pts_per_prediction *= 1.5

            # Get some points to start with
            initial_points, initial_values = self.pick_init_points( self.pts_per_prediction )

            start_time = time.time()

            result = gp_minimize(
                func=self.black_box_function,
                dimensions=self.space,
                n_calls=timing_calls,
                n_random_starts=0,
                x0=initial_points,
                y0=initial_values,
                verbose=False
            )

        if self.work_role == 'master':
            LOGGER.info('Optimization Manager: Starting Optimization')
            # Perform the optimization
            result = gp_minimize(
                func=self.black_box_function,
                dimensions=self.space,
                n_calls=n_calls,
                n_random_starts=10,
                x0=initial_points,
                y0=initial_values,
                verbose=True
            )

            return result

    def timing_bayesian_optimization(self, starting_points=10, time_cutoff_sec=10):

        # First, estimate time per task
        start_time = time.time()

        # get random points in space to calc execution time
        n_pts = starting_points
        if self.work_role == 'master':
            n_pts *= self.size - 1

        random_points = []
        for i in range(n_pts):
            x = []
            for dim in self.space:
                x.append( dim.rvs() )
            random_points.append(x)

        # Evaluate the random points
        new_pts, new_val = self._evaluate(random_points)

        # Get time per task
        time_per_task = (time.time() - start_time) / n_pts


        # Use random points with arbitrary values to predict time per prediction and time per model training

        # Initialize the optimizer
        self.optimizer = Optimizer(dimensions=self.space, base_estimator='GP', random_state=42)

        # Generate initial points from what's in the explored points
        initial_points = self.explored_points.tolist()
        initial_values = self.explored_values.tolist()

        # Provide initial data to the optimizer
        self.optimizer.tell(initial_points, initial_values)

        # Lists to store timing data
        predict_times = []
        train_times = []

        current_num_points = int(n_pts)

        # Loop while the prediction step is below the cutoff time
        c=0
        time_tell = True
        time_ask = True
        while True:

            LOGGER.info(f"Timing Bayesian Optimization: Iteration {c} - Num Points: {current_num_points}")

            new_points = np.array([[dim.rvs() for dim in self.space] for _ in range(current_num_points)]).reshape(-1, len(self.space)).tolist()
            new_values = [ np.random.rand() for _ in range(current_num_points)]

            # Set condition for breaking timing if too large
            if time_tell:
                start_time = time.time()
                self.optimizer.tell(new_points, new_values)
                train_time = time.time() - start_time
                train_times.append((current_num_points, train_time))
                LOGGER.info(f"Training Time {c}: {train_time}")
                if train_time > time_cutoff_sec:
                    time_tell = False

            # Step 2: Predict (Ask) step
            if time_ask:
                start_time = time.time()
                predict_n_pts = 1
                _ = self.optimizer.ask(n_points=predict_n_pts)
                predict_time = (time.time() - start_time) # To get time for a single prediction
                predict_times.append((current_num_points, predict_time))
                LOGGER.info(f"Prediction Time {c}: {predict_time}")
                if predict_time > time_cutoff_sec:
                    time_ask = False

            # Break if either time step exceeds the cutoff time
            if train_time > time_cutoff_sec or predict_time > time_cutoff_sec:
                break

            # Double the number of points for the next iteration
            current_num_points = int( 2.0 * current_num_points )
            c += 1
        
        # Assuming quadratic and cubic models for time complexity of prediction and training, get coefficients for models
        predict_times = np.array(predict_times)
        predict_c = np.polyfit(predict_times[:, 0], predict_times[:, 1], deg=2)

        train_times = np.array(train_times)
        train_c = np.polyfit(train_times[:, 0], train_times[:, 1], deg=3)

        self.p_c = predict_c
        self.t_c = train_c

        # print the models and cooefficients
        LOGGER.info(f"Training Time Model: {train_c}")
        LOGGER.info(f"Training Points: \n{train_times}")

        LOGGER.info(f"Prediction Time Model (per {predict_n_pts} pts): {predict_c}")
        LOGGER.info(f"Prediction Points (per {predict_n_pts} pts): \n{predict_times}")
        

        # Create plot showing these times
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.scatter(predict_times[:, 0], predict_times[:, 1], label='Prediction Time (s/{predict_n_pts} pts)', marker='o')
        plt.scatter(train_times[:, 0], train_times[:, 1], label='Training Time (s)', marker='o')
        plt.xlabel('# of PTS in Surrogate Model')

        # Plot the prediction and training time models
        x = np.linspace(int(n_pts), max(predict_times[-1, 0], train_times[-1, 0]), 100)
        plt.plot(x, np.polyval(predict_c, x), label='Prediction Time Model (s/{predict_n_pts} pts)', linestyle='--')
        plt.plot(x, np.polyval(train_c, x), label='Training Time Model', linestyle='--')

        plt.legend()
        plt.title('Timing Analysis of Bayesian Optimization')
        plt.show()

        # Save the plot
        plt.savefig('timing_analysis.png')

        return [ predict_times, predict_c, train_times, train_c ]

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def gen_x_bounds( n_dim = 2, x_range = [-5.12, 5.12] ):
    x = {}
    for i in range(n_dim):
        x[f'x_{i}'] = {'bounds': x_range}
    return x

def test_optimization_manager():
    print("\nTESTING OPTIMIZATION MANAGER\n")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    save_loc = 'test_pts.csv'

    # NOTE: Workers will remain here until manager terminates workers
    if rank != 0:
        worker = Optimization_Manager( rastrigin, gen_x_bounds(2), save_loc=save_loc )
        # worker.run_worker()

    if rank == 0:
        manager = Optimization_Manager( rastrigin, gen_x_bounds(2), save_loc=save_loc )
        # Test gridsearch/exploration

        explored_ar = manager.initial_exploration( time_cutoff_sec=10, timing_points=4 )
        print( "Explored Points: \n", explored_ar )


        # Get timing data for Bayesian Optimization
        # timing_data = opt_manager.timing_bayesian_optimization( time_cutoff_sec=300 )

# If script is called main
if __name__ == '__main__':

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("\OPTIMIZATION MANAGER\n")
        gu.tabprint("This module is intended to be imported and used in other scripts.")
        gu.tabprint("Calling as main will run tests and examples.")
        print("\nInitializing Arguments, and Logger\n")

    try:
        args, LOGGER = gu.initialize_environment()
        # gu.change_logging_level('DEBUG')

        # mpi_logger = logging.getLogger('qm')
        # mpi_logger.setLevel(logging.INFO)

        if rank == 0:
            print( f"\nArgs: \n{args}")
            print( f"\nLogger: \n{LOGGER}")
        
            print("\nChanging LOGGER to Debug")

    except:
        print("Failed to Initalize Arguments and Logger")
        sys.exit(1)

    test_optimization_manager()

