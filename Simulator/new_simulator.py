#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: new_simulator.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden 
Created: 2019-May-10

Description: This script is designed to handle calling the SPAM fortran simulator

References:  Sections of this code were written with the assistance 
    of ChatGPT made by OpenAI.

"""
# ================================ IMPORTS ================================ #

# Standard library imports
import logging, os, sys
import numpy as np
import matplotlib.pyplot as plt


# Add main project directory based on current script location.
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))  # Grab file location
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY) # project directory one layer up
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu
import utilities.model_manager as mm
import custom_runs

# ================================= GLOBALS ================================= #
#   Standardized parameter array for SPAM model:
spam_param_description = '''
    [0]: X-coordinate of the secondary galaxy's position
    [1]: Y-coordinate of the secondary galaxy's position
    [2]: Z-coordinate of the secondary galaxy's position
    [3]: X-component of the secondary galaxy's velocity
    [4]: Y-component of the secondary galaxy's velocity
    [5]: Z-component of the secondary galaxy's velocity
    [6]: Mass of the primary galaxy
    [7]: Mass of the secondary galaxy
    [8]: Outer radius of the primary galaxy
    [9]: Outer radius of the secondary galaxy
    [10]: Azimuthal angle for the primary galaxy
    [11]: Azimuthal angle for the secondary galaxy
    [12]: Inclination angle for the primary galaxy
    [13]: Inclination angle for the secondary galaxy
    [14]: Softening length for the primary galaxy
    [15]: Softening length for the secondary galaxy
    [16]: Scaling factor for bulge in the primary galaxy
    [17]: Scaling factor for disk in the primary galaxy
    [18]: Scaling factor for halo in the primary galaxy
    [19]: Scaling factor for bulge in the secondary galaxy
    [20]: Scaling factor for disk in the secondary galaxy
    [21]: Scaling factor for halo in the secondary galaxy
'''

 # Arbitrary limit in case of typo, can be changed if needed
MAX_PARTICLE_COUNT = 1e5

# Global logger from main program
LOGGER = logging.getLogger(__name__)

# ================================= CORE FUNCTIONS ================================= #

def basic_disk_wrapper( collision_param, npts1 = 100, npts2 = 50, dynamic_friction_lnl = 0.001):
    """
    This function is a wrapper for the custom_runs_module.basic_disk function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
        dynamic_friction_lnl (float): Variable to adjust strength of dynamic friction
    
    Returns:
        disk_pts (np.ndarray): Array of particles for primary and secondary disk

    """

    # ensure array is in the correct format for fortran
    f_ar = np.asfortranarray(np.array(collision_param).astype(np.float64))

    # Call the Fortran function
    LOGGER.debug(f"Calling custom_runs.basic_disk: {collision_param}")
    try:
        disk_pts = custom_runs.custom_runs_module.basic_disk( f_ar, npts1, npts2, dynamic_friction_lnl )
    except:
        LOGGER.error(f"Failed to call custom_runs.basic_disk")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"npts1 - npts2: {npts1} - {npts2}")
        raise ValueError(f"Failed to call custom_runs.basic_disk")
    
    LOGGER.debug(f"Returned Disk Particles: {disk_pts.shape}")

    return disk_pts

def orbit_run_wrapper( collision_param, dynamic_friction_lnl = 0.001 ):
    """
    This functins is a warpper for the custom_runs_module.orbit_run function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        dynamic_friction_lnl (float): Variable to adjust strength of dynamic friction

    Returns:
        orbit_path (np.ndarray): Array of particles indicating the path of the secondary galaxy
                        NOTE: The primary galaxy is fixed at origin throughout the simulation.
    """

    # ensure array is in the correct value format
    in_ar = np.array(collision_param).astype(np.float64)

    # Convert the array to Fortran contiguous
    in_ar = np.asfortranarray(in_ar)

    # We need to know how large the final orbit path will be before calling the function
    LOGGER.debug(f"Calling custom_runs.calc_orbit_time_steps: {collision_param}")
    try:
        n_time_steps = custom_runs.custom_runs_module.calc_orbit_integration_steps( in_ar, dynamic_friction_lnl )

    except:
        LOGGER.error(f"Failed to call 'custom_runs.calc_orbit_time_steps'")
        raise ValueError(f"Failed to call 'custom_runs.calc_orbit_time_steps'")
    
    # Call the Fortran function to create the orbit path
    LOGGER.debug(f"Calling custom_runs.orbit_run.  Time steps: {n_time_steps}")
    try:
        orbit_path = custom_runs.custom_runs_module.orbit_run( in_ar, n_time_steps, dynamic_friction_lnl )
    except:
        LOGGER.error(f"Failed to call 'custom_runs.orbit_run'")
        raise ValueError(f"Failed to call 'custom_runs.orbit_run'")

    LOGGER.debug(f"Returned Orbit Path: {orbit_path.shape}")

    return orbit_path

def basic_run_wrapper( collision_param, npts1 = 100, npts2 = 50, heat1 = 0.0, heat2 = 0.0, dynamic_friction_lnl = 0.001):
    """
    This function is a wrapper for the custom_runs_module.basic_disk function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
        heat1 (float): random motion parameter for the primary galaxy
        heat2 (float): random motion parameter for the secondary galaxy
        dynamic_friction_lnl (float): Variable to adjust strength of dynamic friction
    
    Returns:
        init_pts (np.ndarray): Array of particles before collision interactions
        final_pts (np.ndarray): Array of particles after collision interactions

    """
    
    # ensure array is in the correct value format
    in_ar = np.array(collision_param).astype(np.float64)

    # Convert the array to Fortran contiguous
    in_ar = np.asfortranarray(in_ar)

    # Call the Fortran function
    LOGGER.debug(f"Calling custom_runs.basic_disk: {collision_param}")
    try:
        init_pts, final_pts = custom_runs.custom_runs_module.basic_run( in_ar, npts1, npts2, heat1, heat2, dynamic_friction_lnl )
    except:
        LOGGER.error(f"Failed to call 'custom_runs.basic_run'")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"npts1 - npts2: {npts1} - {npts2}")
        LOGGER.error(f"heat1 - heat2: {heat1} - {heat2}")
        raise ValueError(f"Failed to call 'custom_runs.basic_run'")
    
    LOGGER.debug(f"Returned Disk Particles: {init_pts.shape} - {final_pts.shape}")

    return (init_pts, final_pts)

# ======================== PLOT FUNCTIONS ======================== #

def plot_all_from_param(collision_param, plot_loc=None):
    """
    This function will take collision_parameters and plot several aspects of the simulation

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        plot_loc (str, optional): Location to save the plot

    Returns:
        plot_fig (matplotlib.figure.Figure): Figure with several subplots
    """

    LOGGER.info("Plotting Orbit of a Simulation")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    n1 = 500
    n2 = 500

    # Call functions to get orbit and particle data
    orbit_path = orbit_run_wrapper(collision_param)
    init_pts, final_pts = basic_run_wrapper(collision_param, n1, n2)

    # Calculate index of minimum radius to origin (Distance of closest approach)
    min_r_idx = np.argmin(np.linalg.norm(orbit_path, axis=1))

    # Function to calculate the normal vector of the orbit plane
    def plane_normal(points):
        # Calculate the covariance matrix of the points
        cov_matrix = np.cov(points.T)
        # Eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        # The eigenvector with the smallest eigenvalue is the normal vector of the plane
        normal_vector = eigvecs[:, np.argmin(eigvals)]
        return normal_vector

    # Calculate the normal vector of the orbit path
    normal_vector = plane_normal(orbit_path)

    # Convert the normal vector to spherical coordinates for setting the viewing angle
    r = np.linalg.norm(normal_vector)
    elev = np.degrees(np.arccos(normal_vector[2] / r))
    azim = np.degrees(np.arctan2(normal_vector[1], normal_vector[0]))

    # Create figure with top and bottom subplots    # Create figure with top and bottom subplots
    fig = plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[2, 2, 2, 2, 2, ])

    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2:], projection='3d')

    # FIRST SUBPLOT: Final particles as seen from perspective
    ax1.scatter(final_pts[0:n1, 0], final_pts[0:n1, 1], s=1, c='b', label='Primary Disk')
    ax1.scatter(final_pts[n1:n1+n2, 0], final_pts[n1:n1+n2, 1], s=1, c='r', label='Secondary Disk')
    ax1.scatter(final_pts[-1, 0], final_pts[-1, 1], s=10, c='k', label=f'Secondary Center: ({final_pts[-1,0]:.2f},{final_pts[-1,1]:.2f})')
    ax1.set_title('Final Particles Perspective')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.set_aspect('equal', 'box')

    # SECOND SUBPLOT: 3D plot with good angle of orbit path
    ax2.scatter(final_pts[::3, 0], final_pts[::3, 1], final_pts[::3, 2], c='r', label='Final')
    ax2.scatter(init_pts[::3, 0], init_pts[::3, 1], init_pts[::3, 2], c='b', label='Initial')
    ax2.plot(orbit_path[:, 0], orbit_path[:, 1], orbit_path[:, 2], c='k', label='Orbit Path')
    ax2.quiver(0, 0, 0, orbit_path[min_r_idx, 0], orbit_path[min_r_idx, 1], orbit_path[min_r_idx, 2], color='k', label='Min Radius Vector')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Orbit Path')
    ax2.legend()
    ax2.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    # If plot_loc is given, save the figure to that location
    if plot_loc:
        plt.savefig(plot_loc)

    return fig


# ======================== MAIN / TESTING ========================== #

if __name__ == '__main__':

    print("\SIMULATOR\n")
    gu.tabprint("This module is intended to be imported and used in other scripts.")
    gu.tabprint("Running script as main will run tests and examples.")

    print("\nInitializing Arguments, and Logger\n")
    try:
        args, LOGGER = gu.initialize_environment()
    except:
        print("Failed to Initalize Arguments and Logger")
        sys.exit(1)

    print( f"\nArgs: \n{args} \nn")
    print( f"\nLogger: \n{LOGGER}\n")
    
    print("Changing LOGGER to Debug")
    gu.change_logging_level('DEBUG')

    print("\nReimport custom modules since Logger changed\n")
    import importlib
    importlib.reload(mm)

    print("\n Testing model_manager (storage_mode = memory)\n")
    test_param = [-9.93853,-4.5805,3.27377,-0.50008,-2.45565,-1.07799,23.33004,24.69427,3.49825,5.32056,309.6923,36.8125,41.78471,51.42857,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.94594,0,0,0.18901,-22.47999,9.02372,0.0,1.0,0.0,0.0,0.0,0.0][0:22]
    test_manager = mm.Model_Manager( model_data=test_param )
    print( f"\nModel_Manger: \n{test_manager}" )
    print( f"\nManager Model Data: \n{test_manager.info['model_data']}" )

    print("\nTesting basic_disk")
    basic_disk_wrapper( test_param )

    print("\nTesting orbit_run")
    orbit_run_wrapper( test_param )

    print("\nTesting Basic Run")
    ipts, fpts = basic_run_wrapper( test_param )

    print("\nThis SHOULD cause an issue")
    try:
        basic_disk_wrapper( None )
    except:
        print("Failed as expected")

    print("\nPlotting Example Collision")

    print("Changing LOGGER to INFO becuase matplotlib gives way too many debug messages.")
    gu.change_logging_level('INFO')
    import importlib
    importlib.reload(mm)

    plot_all_from_param( test_param, 'test_orbit_plot.png' )

    print()
    