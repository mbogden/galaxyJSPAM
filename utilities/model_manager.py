#!/usr/bin/env python
# coding: utf-8

"""
File: model_manager.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2020-Feb-21

Description: This code is designed to abstract galactic encounter models and how their information is stored and accessed.

References:  Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.

"""

# ================================ IMPORTS ================================ #
# Standard library imports
import logging
import os
import sys

# Third-party imports
import cv2


# Add main project directory to import project modules
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import utilities.general_utility as gu

# ================================ GLOBAL VARIABLES ================================ #
LOGGER = logging.getLogger()

# ================================ CLASSES ================================ #

class Model_Manager:
    """
    Class to manage information and operations for galactic encounter models (runs).
    """

    def __init__(self, model_data, model_directory = None, model_id = None, storage = 'memory', new_model = False ):
        """
        Initializes the Model_Manager instance.

        Parameters:
            model_data (numpy.ar): The data of the model. Defaults to None.
            model_directory (str, optional): The directory where the model is stored. Defaults to None.
            model_id (str, optional): The unique identifier of the model. Defaults to None.
            storage (str, optional): Type of storage to use ('memory' or 'disk'). Defaults to 'memory'.
            new_model (bool, optional): Flag to indicate if this is a new model. Defaults to False.

        Raises:
            ValueError: If any required parameter is None or invalid.
        """

        LOGGER.debug(f"Initializing Model_Manager")

        # Store model info
        self.model_directory = model_directory
        self.info = {}
        self.info['model_data'] = model_data
        self.info['model_id'] = model_id

        # New model use case, keep data only in memory
        if storage == 'memory':  
            LOGGER.debug(f"Using memory only")

            # verify model data is not None
            if self.info['model_data'] is None:
                LOGGER.error(f"Model data is None")
                raise ValueError(f"Model data is None")
            
            # Indicate manager is not setup for disk storage and return
            self.disk_status = False
            return

        # Should only reach this point if saving model to disk storage, 

        # Verify storage is set to disk
        if storage != 'disk':
            LOGGER.error(f"Invalid storage type: {storage}")
            raise ValueError(f"Invalid storage type: {storage}")

        # setup directory and files for disk storage
        LOGGER.debug(f"Using disk storage")

        # Verify model directory is string
        if not isinstance(self.model_directory, str):
            LOGGER.error(f"Model directory is not a string: {model_directory}")
            raise ValueError(f"Model directory is not a string: {model_directory}")

        # Create model directory if new model
        if new_model:            
            LOGGER.info(f"Creating new model: {model_id}")

            # Verify model data and model id are not None
            if self.info['model_data'] is None:
                LOGGER.error(f"Model data is None")
                raise ValueError(f"Model data is None")
            
            if self.info['model_id'] is None:
                LOGGER.error(f"Model id is None")
                raise ValueError(f"Model id is None")

            # Create model directory
            LOGGER.info(f"Creating new model directory: {model_directory}")
            os.makedirs(self.model_directory, exist_ok=True)

        # end if new model

        # Verify model directory exists
        self.model_directory = gu.valid_path(self.model_directory)

        # Raise error if not valid
        if self.model_directory is None:
            LOGGER.warning(f"Invalid model directory: {model_directory}")
            raise ValueError(f"Invalid model directory: {model_directory}")

        # Define Directory locations
        self._particle_dir = f"{self.model_directory}/particle_files"
        self._image_dir = f"{self.model_directory}/model_images"
        self._temp_dir = f"{self.model_directory}/temp"

        # Define File locations
        self._info_loc = f"{self.model_directory}/info.json"

        # If new model create directories and files
        if new_model:
            LOGGER.info(f"Creating new model directories and files")
            os.makedirs(self._particle_dir, exist_ok=True)
            os.makedirs(self._image_dir, exist_ok=True)
            os.makedirs(self._temp_dir, exist_ok=True)
            self.save_info()

        # Check if directories and files exist, if not raise error
        required_paths = [self._particle_dir, self._image_dir, self._temp_dir, self._info_loc]
        for path in required_paths:
            if not os.path.exists(path):
                LOGGER.warning(f"Required item does not exist: {path}")
                raise ValueError(f"Required item does not exist: {path}")

        # Read info file
        LOGGER.debug(f"Reading info data: {self._info_loc}")
        self.info = gu.read_json( self._info_loc )

        # Verify model data and model id 
        if self.info['model_data'] is None:
            LOGGER.error(f"Model data is None")
            raise ValueError(f"Model data is None")

        if self.info['model_id'] is None:
            LOGGER.error(f"Model id is None")
            raise ValueError(f"Model id is None")
        
        # Should only reach this point if disk storage is setup for model manager. 
        self.disk_status = True

    def save_info( self,):
        """
        Saves the info dictionary to the info.json file.

        This method serializes the `info` dictionary of the class instance
        and writes it to the `_info_loc` file path in JSON format. This 
        is used to persist the model's state to disk.

        Raises:
            Exception: If there is an error during the writing process.
        """
        LOGGER.debug(f"Saving info data: {type(self.info)} at {self._info_loc}")            
        gu.write_json( self._info_loc, self.info,  pretty=True, convert_numpy_array=True )

    # End save info file

# End Model_Manager class

# Testing class 
def test_model_manager():

    global LOGGER

    # Sample SPAM data
    sample_data = [[ 9.29623333e-01,  1.00534000e+00, -9.38000000e-02,  2.79565681e-01,
    -4.54919137e-01,  1.16685366e-01,  1.40585743e+00,  8.31712243e-01,
    1.33617965e+00,  1.24469114e+00, -5.87146308e+01,  1.50859166e+02,
    2.56197792e+01,  1.10851182e+02,  4.45393216e-01,  4.14897047e-01,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00], [ 9.29623333e-01,  1.00534000e+00, -9.38000000e-02,  2.79565681e-01,
    -4.54919137e-01,  1.16685366e-01,  1.41985614e+00,  7.83177914e-01,
    1.31837239e+00,  1.13119519e+00, -6.35853782e+01,  1.48904074e+02,
    2.47226766e+01,  1.13561007e+02,  4.39457464e-01,  3.77065064e-01,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00],]

    # Create Model Manager
    print("Testing Model_Manager with model_data and memory only")

        # Now create an instance of ModelManager or call its methods
    #manager = ModelManager()  # Assuming default constructor or modify as needed
    
    # Example function calls

    try:
        print('The following tests should work')

        print('\nTesting model data with memory only')
        model_manager = Model_Manager( model_data = sample_data[0] )

        print("\nTesting new model with disk storage")
        model_manager = Model_Manager( model_data = sample_data[0], model_directory = 'test_run_dir', model_id = 'test_model', storage='disk', new_model=True )
                                      
        print("\nTesting existing model with disk storage")
        model_manager = Model_Manager( model_directory = 'test_run_dir', storage='disk' )

        print("\nThis should NOT Work")
        model_manager = Model_Manager( model_data = sample_data[0], model_directory = 'test_run_dir', storage='disk', new_model=True )
    
    except Exception as e:
        LOGGER.critical(f"An error occurred: {e}")

# ================================ MAIN ================================ #

# If script is called main
if __name__ == '__main__':

    print("\nMODEL MANAGER\n")
    gu.tabprint("This module is intended to be imported and used in other scripts.")
    gu.tabprint("Calling as main will run tests and examples.")

    print("\nInitializing Arguments, and Logger\n")
    try:
        args, LOGGER = gu.initialize_environment()
    except:
        print("Failed to Initalize Arguments and Logger")
        sys.exit(1)

    print( f"\nArgs: \n{args}")
    print( f"\nLogger: \n{LOGGER}")
    
    print("\nChanging LOGGER to Debug")
    gu.change_logging_level('DEBUG')

    print("\nTESTING LOGGER\n")
    LOGGER.debug("Debug Test")
    LOGGER.info("Info Test")
    LOGGER.warning("Warning Test")
    LOGGER.error("Error Test")
    LOGGER.critical("Critical Test")

    test_model_manager()