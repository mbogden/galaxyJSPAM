#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# Add main project directory based on current script location.
try:
    SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))  # Grab file location
except: 
    SCRIPT_DIRECTORY = os.getcwd()  # Assume that the script is being run from where it's located

PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
sys.path.append(PROJECT_DIRECTORY)

# import project modules
import general_utility as gu


# In[2]:


# ================================ VARIABLES ================================ #
JUP_ENV = gu.in_jupyter_notebook()   # Change this as you wish for testing and trouble shooting

if JUP_ENV:
    LOGGER = gu.configure_logging()
else:
    LOGGER = logging.getLogger(__name__)


# In[23]:


# ================================ CLASSES ================================ #

class Model_Manager:
    """
    Class to manage information and operations for galactic encounter models (runs).
    """

    def __init__(self, model_data = None, model_directory = None, model_id = None, storage = 'memory', new_model = False ):
        
        LOGGER.info(f"Initializing Model_Manager")

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
                LOGGER.warning(f"Model data is None")
                raise ValueError(f"Model data is None")
            
            # Indicate manager is not setup for disk storage and return
            self.disk_status = False
            return

        # Should only reach this point if saving model to disk storage, 

        # Verify storage is set to disk
        if storage != 'disk':
            LOGGER.warning(f"Invalid storage type: {storage}")
            raise ValueError(f"Invalid storage type: {storage}")

        # setup directory and files for disk storage
        LOGGER.debug(f"Using disk storage")

        # Verify model directory is string
        if not isinstance(self.model_directory, str):
            LOGGER.warning(f"Model directory is not a string: {model_directory}")
            raise ValueError(f"Model directory is not a string: {model_directory}")

        # Create model directory if new model
        if new_model:            
            LOGGER.info(f"Creating new model: {model_id}")

            # Verify model data and model id are not None
            if self.info['model_data'] is None:
                LOGGER.warning(f"Model data is None")
                raise ValueError(f"Model data is None")
            
            if self.info['model_id'] is None:
                LOGGER.warning(f"Model id is None")
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
            LOGGER.warning(f"Model data is None")
            raise ValueError(f"Model data is None")

        if self.info['model_id'] is None:
            LOGGER.warning(f"Model id is None")
            raise ValueError(f"Model id is None")
        
        # Should only reach this point if disk storage is setup for model manager. 
        self.disk_status = True

    def save_info( self,):
        """
        Save the info dictionary to the info file.
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

    gu.change_logging_level('DEBUG')
    try:
        print('The following tests should work')

        print('Testing model data with memory only')
        model_manager = Model_Manager( model_data = sample_data[0] )

        print("Testing new model with disk storage")
        model_manager = Model_Manager( model_data = sample_data[0], model_directory = 'tmp_run_dir', model_id = 'tmp_model', storage='disk', new_model=True )
                                      
        print("Testing existing model with disk storage")
        model_manager = Model_Manager( model_directory = 'tmp_run_dir', storage='disk' )

        # model_manager = Model_Manager( model_data = sample_data[0], storage='disk' )
        model_manager = Model_Manager( model_data = sample_data[0], model_directory = 'tmp_run_dir', storage='disk', new_model=True )
        # model_manager = Model_Manager( model_data = sample_data[0], model_directory = 'tmp_run_dir', storage='disk' )
    except Exception as e:
        print(f"An error occurred: {e}")


if JUP_ENV and True:
    test_model_manager()


# In[34]:


# If main

def main( args = None ):

    global LOGGER

    # init args and logger
    try:
        args, LOGGER = gu.initialize_environment(args=args)
    except:
        print("Failed to Initalize Arguments and Logger")
        return


    # Run tests if requeted
    if getattr( args, 'run_tests', False ):
        test_model_manager()

if __name__ == '__main__':
    main()


# In[ ]:




