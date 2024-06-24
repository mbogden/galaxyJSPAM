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

def basic_disk_wrapper( collision_param, npts1 = 100, npts2 = 50, heat1 = 0.0, heat2 = 0.0 ):
    """
    This function is a wrapper for the custom_runs_module.basic_disk function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
        heat1 (float): random motion parameter for the primary galaxy
        heat2 (float): random motion parameter for the secondary galaxy
    
    Returns:
        disk_pts (np.ndarray): Array of particles for primary and secondary disk

    """

    # ensure array is in the correct value format
    f_ar = np.array(collision_param).astype(np.float64)

    # Convert the array to Fortran contiguous
    f_ar = np.asfortranarray(collision_param)

    # Call the Fortran function
    LOGGER.debug(f"Calling custom_runs.basic_disk: {collision_param}")
    try:
        disk_pts = custom_runs.custom_runs_module.basic_disk( f_ar, npts1, npts2, heat1, heat2 )
    except:
        LOGGER.error(f"Failed to call custom_runs.basic_disk")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"npts1 - npts2: {npts1} - {npts2}")
        LOGGER.error(f"heat1 - heat2: {heat1} - {heat2}")
        raise ValueError(f"Failed to call custom_runs.basic_disk")
    
    LOGGER.debug(f"Returned Disk Particles: {disk_pts.shape}")

    return disk_pts


def basic_run_wrapper( collision_param, npts1 = 100, npts2 = 50, heat1 = 0.0, heat2 = 0.0 ):
    """
    This function is a wrapper for the custom_runs_module.basic_disk function.

    Parameters:
        collision_param (np.ndarray): Array of collision parameters
        npt1 (int): Number of particles for the primary galaxy
        npt2 (int): Number of particles for the secondary galaxy
        heat1 (float): random motion parameter for the primary galaxy
        heat2 (float): random motion parameter for the secondary galaxy
    
    Returns:
        init_pts (np.ndarray): Array of particles before collision interactions
        final_pts (np.ndarray): Array of particles after collision interactions

    """

    # ensure array is in the correct value format
    in_ar = np.array(collision_param).astype(np.float64)

    # Convert the array to Fortran contiguous
    in_ar = np.asfortranarray(collision_param)

    # Call the Fortran function
    LOGGER.debug(f"Calling custom_runs.basic_disk: {collision_param}")
    try:
        init_pts, final_pts = custom_runs.custom_runs_module.basic_run( in_ar, npts1, npts2, heat1, heat2 )
    except:
        LOGGER.error(f"Failed to call 'custom_runs.basic_run'")
        LOGGER.error(f"Collision Param: {collision_param}")
        LOGGER.error(f"npts1 - npts2: {npts1} - {npts2}")
        LOGGER.error(f"heat1 - heat2: {heat1} - {heat2}")
        raise ValueError(f"Failed to call 'custom_runs.basic_run'")
    
    LOGGER.debug(f"Returned Disk Particles: {init_pts.shape} - {final_pts.shape}")

    return (init_pts, final_pts)


# ======================== MAIN / TESTING ========================== #

if __name__ == '__main__':

    print("\nNEW SIMULATOR\n")
    gu.tabprint("This module is intended to be imported and used in other scripts.")
    gu.tabprint("Calling as main will run tests and examples.")

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

    print("\nTESTING LOGGER\n")
    LOGGER.debug("Debug Test")
    LOGGER.info("Info Test")
    LOGGER.warning("Warning Test")
    LOGGER.error("Error Test")
    LOGGER.critical("Critical Test")

    print("\nReimport custom modules since Logger changed\n")
    import importlib
    importlib.reload(mm)

    print("\n Testing model_manager (storage_mode = memory)\n")
    test_param = [-9.93853,-4.5805,3.27377,-0.50008,-2.45565,-1.07799,23.33004,24.69427,3.49825,5.32056,309.6923,36.8125,41.78471,51.42857,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.94594,0,0,0.18901,-22.47999,9.02372,0.0,1.0,0.0,0.0,0.0,0.0][0:22]
    test_manager = mm.Model_Manager( model_data=test_param)
    print( f"\nModel_Manger: \n{test_manager}" )
    print( f"\nManager Model Data: \n{test_manager.info['model_data']}" )

    print("\nTesting basic_disk")
    basic_disk_wrapper( test_param)

    print("\nTesting Basic Run")
    basic_run_wrapper( test_param )

    print("\nThis SHOULD cause an issue")
    try:
        basic_disk_wrapper( None )
    except:
        print("Failed as expected")


    