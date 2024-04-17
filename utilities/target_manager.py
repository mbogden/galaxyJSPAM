#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: target_manager.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2020-Feb-21

Description: This code is designed to abstract galactic encounter targets and how their information is stored and accessed.

References:  Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.

"""
# ================================ IMPORTS ================================ #
# Standard library imports
import logging
import os
import sys

# Third-party imports

try:
    # Add main project directory based on current script location.
    script_directory = os.path.dirname(os.path.realpath(__file__))
    project_directory = os.path.dirname(script_directory)
    sys.path.append(project_directory)

except: # Assume that the script is being run from where it's located
    script_directory = os.getcwd()
    project_directory = os.path.dirname(script_directory)
    sys.path.append(project_directory)


# import general_utility and initial logger
import general_utility as gu

logger = logging.getLogger(__name__)




# In[ ]:


class TargetManager:
    """
    Class to manage information and operations related to targets in the Galaxy Zoo: Merger project.
    """

    def __init__(self, target_dir=None, args=None):

        logger.info("Initializing TargetManager")
        
        self.target_dir = target_dir if target_dir else args.target_dir if args else None
        assert self.target_dir, "Target directory must be specified."

        self.init_paths()
        self.configure_directories()

        if args and getattr(args, 'new_target', False):
            self.setup_new_target(args)

        self.load_target_info()
        self.status = self.check_integrity()

        function __init__(targetDir, tArg, printBase, printAll):


    #     if not target_dir:
    #         logging.error("Target directory is not specified.")
    #         raise ValueError("Target directory must be specified.")

    #     self.target_dir = target_dir
    #     self.init_paths()

    #     try:
    #         self.validate_directories()
    #     except Exception as e:
    #         logging.error(f"Failed to validate directories: {e}")
    #         raise

    #     try:
    #         self.load_target_info()
    #     except FileNotFoundError:
    #         logging.error("Target info file not found.")
    #         raise

    #     logging.info("TargetManager initialized successfully.")

    # def init_paths(self):
    #     # Initialize paths
    #     self.info_dir = os.path.join(self.target_dir, 'information')
    #     logging.debug(f"Information directory set to {self.info_dir}")

    # def validate_directories(self):
    #     # Check required directories exist
    #     if not os.path.exists(self.info_dir):
    #         logging.error(f"Information directory does not exist: {self.info_dir}")
    #         raise FileNotFoundError(f"Information directory does not exist: {self.info_dir}")

    # def load_target_info(self):
    #     # Load or initialize the target information
    #     info_path = os.path.join(self.info_dir, 'target_info.json')
    #     if not os.path.exists(info_path):
    #         logging.error(f"No target info found at {info_path}")
    #         raise FileNotFoundError(f"No target info found at {info_path}")
    #     logging.info("Target info loaded successfully.")


        # if tArg is None:
        #     create default tArg
        # set printing preferences (printBase, printAll)
        # if printing detailed info:
        #     print initial setup details
        # if target directory setup in tArg is incorrect:
        #     handle error (print and exit)
        # if a new target is being set up:
        #     setup new target (directories, base files)
        # check and prepare directory structure
        # try to load target information from JSON
        # if score CSV exists:
        #     load scores into DataFrame
        # update status to True (indicates successful setup)


    def init_paths(self):
        """
        Initialize the paths to common directories and files used by the manager.
        """
        self.info_dir  = f'{self.target_dir}/information/'
        self.gzm_dir   = f'{self.target_dir}/galaxy_zoo_merger_models/'
        self.plot_dir  = f'{self.target_dir}/plots/'
        self.image_dir = f'{self.target_dir}/target_images/'
        self.tmp_dir   = f'{self.target_dir}/tmp/'
        self.param_dir = f'{self.target_dir}/score_parameters/'

        self.all_info_loc  = f'{self.target_dir}/target_info.json'
        self.base_info_loc = f'{self.info_dir}/base_target_info.json'
        self.score_loc     = f'{self.target_dir}/scores.csv'


    def configure_directories(self):
        """
        Ensure all necessary directories are present and create them if not.
        """
        logger.debug(f"Configuring directories at {self.target_dir}")
        required_dirs = [self.info_dir, self.gzm_dir, self.tmp_dir, self.plot_dir,
                         self.image_dir, self.param_dir]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)


    def setup_new_target(self, args):
        """
        Set up a new target, initializing directories and base files as needed.
        """
        logger.warning(f"Configuring directories at {self.target_dir}")

        import shutil

        # if reset-target is specified, remove some directories and files
        if getattr(args, 'reset-target', False):

            rm_paths = [self.gzm_dir, self.tmp_dir, self.plot_dir,
                    self.image_dir, self.param_dir, self.all_info_loc, self.score_loc]

            for path in rm_paths:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)             

            self.configure_directories()
        
        # Get target name from args
        getattr(args, 'target-name', False)



    def load_target_info(self):
        """
        Load or initialize the target information from a JSON file.
        """
        import json
        try:
            with open(self.all_info_loc, 'r') as file:
                self.target_info = json.load(file)
        except FileNotFoundError:
            self.target_info = {}
            if self.verbose:
                print(f"No existing target info, initialized new at {self.all_info_loc}")

    def check_integrity(self):
        """
        Check the integrity of the loaded or initialized information.
        """
        required_keys = ['target_id', 'target_images', 'model_data']
        if all(key in self.target_info for key in required_keys):
            return True
        if self.verbose:
            print("Target information is incomplete or corrupted.")
        return False


# Set main function to run if script is run
if __name__ == '__main__':

    # Use GU arg manager
    tArgs = gu.ArgHandler()
    tMan = TargetManager(tArgs)
    pass

