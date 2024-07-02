#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: general_utility.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden 
Created: 2020-Feb-28

Description: This script is designed to handle many common functions 
    within the galacticConstraints project.

References:  Sections of this code were rewritten with the assistance 
    of ChatGPT made by OpenAI.

"""
# ================================ IMPORTS ================================ #

# Standard library imports
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pprint import PrettyPrinter

# Third-party imports
import cv2
import numpy as np
from mpi4py import MPI


# ================================= GLOBALS ================================= #
# Initialize MPI environment
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


# ======================== PARSING INPUT ARGUMENTS ========================== #

class ArgHandler:
    """
    Handles command-line arguments for the galacticConstraints application.
    
    This class uses argparse to parse command-line arguments and can also load additional 
    arguments from a specified configuration file. The class allows for overriding specific 
    arguments via the command line, providing flexibility in application configuration.
    
    Attributes:
        args (Namespace): A namespace populated with arguments obtained from the command line 
                          and configuration file.
    """

    def __init__(self, args=None):        
        """
        Initializes the ArgHandler instance by parsing command-line arguments
        and loading additional settings from a configuration file if specified.
        """
        self.args = self.parse_args(args)

    def parse_args(self, args=None):
        """
        Parses command-line arguments and loads additional settings from a 
        configuration file if specified via the --config argument.
        
        Returns:
            An argparse.Namespace object populated with application arguments.
        """
        parser = argparse.ArgumentParser(
            description='Process and handle various operations for simulation, image creation, and scoring.',
            epilog='''
                    Common Commands:

                    Initialize a brand new data directory for all galaxy_zoo_merger targets:
                    python3 main_simr.py --data-dir path/to/data/directory/ --new-info --new-base --new-run-info --new-run-base

                    Tell program to point at a target directory and generate new machine scores from a parameter file you've created:
                    python3 main_simr.py --target-dir path/to/target/ --new-all --score-param-loc path/to/param.json

                    Tell program to focus on a single model, create a new image, overwrite existing images and print all progress. Useful for testing new image creation:
                    python3 main_simr.py --print-all --run-dir path/to/run/ --new-image --overwrite --score-param-loc path/to/new_param.json

                    Tell program to go through many targets creating new simulations, images, and score as needed. Use 24 processors on current machine:
                    mpirun -n 24 python3 main_simr.py --data-dir path/to/all/targets/ --new-all --score-param-loc path/to/param.json

                    Test Genetic Algorithm Experiment using testing parameters:
                    mpirun -n 8 python3 main_simr.py --target-dir path/to/target --score-param-loc path/to/exp_score.json --print-all --ga-exp --ga-param-loc param/exp_ga.json
                    ''',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument('--config', type=str, help='Path to configuration file containing program arguments')
        parser.add_argument('--new-config', action='store_true', default=False, help='Overwrite config file with current arguments.')
        parser.add_argument('--log-level', type=str, choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], 
                            default='INFO', help='Set logging level for application. Options: CRITICAL, ERROR, WARNING, INFO, and DEBUG. Default is INFO.')
        parser.add_argument('--test', action='store_true', default=False, help='Runs tests for current module.')

        parser.add_argument('--model-dir', type=str, help='Path to a model directory.')
        parser.add_argument('--target-dir', type=str, help='Path to a target directory.')
        parser.add_argument('--data-dir', type=str, help='Path to directory that contains many target directories.')
        
        parser.add_argument('--score-param-loc', type=str, help='Path to an existing score parameter file.')
        parser.add_argument('--score-param-name', type=str, help='Name of a score parameters file already saved in target\'s score_parameters folder.')
        parser.add_argument('--target-loc', type=str, help='Path to an existing target image.')
        
        parser.add_argument('--worker-data-loc', type=str, default='target', help='Where should workers save data.')
        parser.add_argument('--ga-exp', action='store_true', default=False, help='Short for "Genetic Algorithm Experiment".')
        parser.add_argument('--ga-param-loc', type=str, help='Location for genetic algorithm parameter file.')
        
        parser.add_argument('--new-sim', action='store_true', default=False, help='Tells program to create a new simulation of a model.')
        parser.add_argument('--zip-sim', action='store_true', default=False, help='Tells program to zip particle files created by simulator. Warning, files can be large.')
        parser.add_argument('--new-image', action='store_true', default=False, help='Tells program to create a new model image.')
        parser.add_argument('--new-feats', action='store_true', default=False, help='Tells program to create WNDCHRM features out of model image.')
        parser.add_argument('--new-score', action='store_true', default=False, help='Tells program to create a new machine score.')
        parser.add_argument('--new-plot', action='store_true', default=False, help='Tells program to create series of plots for a target.')
        parser.add_argument('--new-all', action='store_true', default=False, help='Tells program to create a new simulation, image, and score.')
        
        parser.add_argument('--overwrite', action='store_true', default=False, help='Tells program to create new files even if they already exist.')
        parser.add_argument('--new-info', action='store_true', default=False, help='Remove current information file. Create a new one from a copy of the base.')
        parser.add_argument('--new-base', action='store_true', default=False, help='Remove the base information file and generate a new one.')
        parser.add_argument('--new-run-info', action='store_true', default=False, help='Will remove the information file for all models found in a target.')
        parser.add_argument('--new-run-base', action='store_true', default=False, help='Will remove the base information file and generate a new one for all models found in a target.')
        parser.add_argument('--start-run', type=int, default=0, help='While iterating through target models, start with N\'th model.')
        parser.add_argument('--end-run', type=int, default=-1, help='While iterating through target models, end with N\'th model. (-1 to go until the end).')
        parser.add_argument('--skip-run', type=int, default=1, help='While iterating through target models, skip to every N\'th model.')
        
        # These have not been used in sometime.  Consider removing
        parser.add_argument('--norm-feats', action='store_true', default=False, help='Takes WNDCHRM features created in runs and normalizes them. Must be paired with -norm-name or -norm-loc.')
        parser.add_argument('--norm-name', type=str, help='Looks for a feature normalization file in target\'s WNDCHRM directory.')
        parser.add_argument('--norm-loc', type=str, help='Looks for feature normalization file in specified path.')
        parser.add_argument('--wndchrm-analysis', action='store_true', help='Performs wndchrm analysis.')


        # Use the provided args or sys.argv[1:] if args is None
        if args is None:  
            args = sys.argv[1:]  # Defaults to sys.argv when args not provided
        parsed_args, unknown = parser.parse_known_args(args)

        # Load and update arguments from the config file
        if parsed_args.config:
            self.load_args_from_file(parsed_args.config, parsed_args)

        # Re-parse to handle any command line overrides
        parsed_args = parser.parse_args(args)

        # Create new config file if requested
        if parsed_args.new_config:
            self.save_args_to_file(parsed_args.config, parsed_args)

        return parsed_args
    
    def load_args_from_file(self, file_path, args):
        """
        Loads arguments from a JSON file and updates the argparse Namespace.
        
        Parameters:
            file_path (str): The path to the configuration file.
            args (Namespace): The existing Namespace object to update with the loaded arguments.
        """
        if not os.path.exists(file_path) and not args.new_config:
            logging.error(f"Configuration file not found: {file_path}")
            return
        elif args.new_config:
            return
                
        with open(file_path, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                # Replace dashes with underscores for attribute names and update the Namespace
                setattr(args, key.replace('-', '_'), value)

    def save_args_to_file(self, file_path, args):
        """
        Saves the current arguments to a JSON file.
        
        Parameters:
            file_path (str): The path where the current arguments should be saved.
        """
        with open(file_path, 'w') as file:
            # Convert argparse Namespace to dict, replacing underscores with dashes in keys
            args_dict = {k.replace('_', '-'): v for k, v in vars(args).items()}
            json.dump(args_dict, file, indent=4)
            logging.info(f"Configuration saved to: {file_path}")

# End ArgHandler class

# ============================ MISC FUNCTIONS ================================ #

# Function to check if running in a Jupyter notebook
def in_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass
    return False

# =========================== LOGGING  ============================ #

# Logging Levels:
# - DEBUG: Detailed information for diagnosing problems (development use).
# - INFO: Confirm things are working as expected (general operation).
# - WARNING: Indicate a potential problem or unexpected event.
# - ERROR: A serious problem that occurred but does not stop the program.
# - CRITICAL: A very serious error suggesting the program might not continue running.
    
def configure_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Configures the logging for the application.

    Sets up the logging format and level. This configuration is intended to be run once at the 
    start of the application to ensure consistent logging behavior across different modules. 
    Allows specifying the logging level to control the verbosity of log messages.

    Parameters:
        log_level (str): The logging level to set for the application as a string 
                         (e.g., 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'). 
                         Defaults to 'INFO'.

    Returns:
        logging.Logger: The configured logger.
    """

    numeric_level = getattr(logging, log_level.upper(), None)
    if numeric_level is None:
        raise ValueError(f'Invalid log level: {log_level}')
    
    log_format = 'Time:%(asctime)s - [%(module)s: %(funcName)s] - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=numeric_level, format=log_format)

    # Add custom filter to include RANK and SIZE
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    logger.filters = []  # Reset filters
    
    return logger

def change_logging_level(log_level):
    """
    Change the logging level dynamically during runtime.

    Parameters:
        log_level (str): The logging level to set (e.g., 'DEBUG', 'WARNING').
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if numeric_level is None:
        raise ValueError(f'Invalid log level: {log_level}')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    logging.info(f'Logging level changed to {log_level}')

# =====   OLD LOGGER FOR JUPYTER NOTEBOOKS   ===== #

# class JupyterNotebookHandler(logging.StreamHandler):

#     def __init__(self):
#         super().__init__()
#         self.setFormatter(logging.Formatter('Time:%(asctime)s - Rank:%(rank)s/%(size)s - [%(module)s: %(funcName)s] - %(levelname)s - %(message)s'))

#     def emit(self, record):
#         from IPython.display import display, HTML
#         message = self.format(record)
#         display(HTML(f'<pre>{message}</pre>'))



# def configure_logging(log_level: str = 'INFO') -> None:
#     """
#     Configures the logging for the application.

#     Sets up the logging format and level. This configuration is intended to be run once at the 
#     start of the application to ensure consistent logging behavior across different modules. 
#     Allows specifying the logging level to control the verbosity of log messages.

#     Parameters:
#         log_level (str): The logging level to set for the application as a string 
#                          (e.g., 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'). 
#                          Defaults to 'INFO'.

#     Returns:
#         None
#     """
#     global RANK, SIZE

#     numeric_level = getattr(logging, log_level.upper(), None)
#     if numeric_level is None:
#         raise ValueError(f'Invalid log level: {log_level}')
    
#     log_format = 'Time:%(asctime)s - Rank:%(rank)s/%(size)s - [%(module)s: %(funcName)s] - %(levelname)s - %(message)s'

#     logging.basicConfig(level=numeric_level, format=log_format)

#     # Add custom filter to include RANK and SIZE
#     logger = logging.getLogger()
#     logger.setLevel(numeric_level)
#     logger.filters = []  # Reset filters


#     # Check if in Jupyter notebook
#     if not in_jupyter_notebook():
#         # Configure standard logging (console or file as needed)
#         logging.basicConfig(level=numeric_level, format=log_format)
    

#     else:
#         # Clear existing handlers and set up for Jupyter Notebook
#         logger.handlers = []  # Clear existing handlers
#         jupyter_handler = JupyterNotebookHandler()
#         logger.addHandler(jupyter_handler)

#     # Add custom filter to include MPI rank and size
#     logger.addFilter(RankSizeFilter(RANK, SIZE))
    
#     return logger

# def change_logging_level(log_level):
#     """
#     Change the logging level dynamically during runtime.

#     Parameters:
#         log_level (str): The logging level to set (e.g., 'DEBUG', 'WARNING').
#     """
#     numeric_level = getattr(logging, log_level.upper(), None)
#     if numeric_level is None:
#         raise ValueError(f'Invalid log level: {log_level}')
#     logger = logging.getLogger()
#     logger.setLevel(numeric_level)
#     logging.info(f'Logging level changed to {log_level}')

# =========================== PRETTY/MISC PRINTING ============================ #

# Pretty Printer Configuration
PP = PrettyPrinter(width=42, compact=True)

def pretty_log(message, log_level):
    """
    Log a message with a specified log level in a pretty format.
    
    Args:
        message (str): The message to log.
        log_level (str): The level at which to log the message.
    """
    pretty_message = pp.pprint.pformat(message)
    if log_level == 'debug':
        logging.debug(message)
    elif log_level == 'info':
        logging.info(message)
    elif log_level == 'warning':
        logging.warning(message)
    elif log_level == 'error':
        logging.error(message)
    elif log_level == 'critical':
        logging.critical(message)
    else:
        logging.error(f'Invalid log level: {log_level}')

def hello():
    """
    Print a greeting message from Matthew's utility module.
    """
    logging.debug("In hello function.")
    print("Hi! You're in Matthew's module for generally useful functions and classes.")

def tabprint(message, begin='\t - ', end='\n'):
    """
    Print a message in a tabbed format.
    
    Args:
        message (str): The message to print.
        begin (str, optional): The prefix for the message. Defaults to '\t - '.
        end (str, optional): The suffix for the message. Defaults to '\n'.
    """
    print(f'{begin}{message}', end=end)

# =========================== INITIALIZE APPLIATION ============================ #

def initialize_environment( args=None ):
    """
    Whichever script is running this function should be main and have its 
    arguments parsed and logging configured for all modules.
    
    Args:

    Return:
        args (Namespace): The parsed arguments.
    """
    configure_logging()
    arg_handler = ArgHandler(args=args)
    logger = logging.getLogger()
    logger.setLevel( arg_handler.args.log_level )
    return arg_handler.args, logger


# ========================= READING/WRITING DISK =========+================== #

def valid_path(input_path, new_path=False):
    """
    Verifies if a path/file exists and returns the full file path.
    
    Args:
        input_path (str): The path to validate.
        log_warning (bool, optional): Whether to log a warning if the path is invalid. Defaults to False.
        
    Returns:
        The absolute path as a string if valid, None otherwise.
    """
    logging.debug("Validating path: " + input_path)
    # Check if input is a string
    if not isinstance(input_path, str):        
        logging.info('Path is not a string')
        return None

    # Check if path exists
    if not os.path.exists(input_path):
        logging.warning('Path does not exist: %s' % input_path)
        return None

    out_path = os.path.abspath(input_path)

    # Normalize directory paths
    if os.path.isdir(out_path) and not out_path.endswith(os.sep):
        out_path += os.sep

    return out_path

def read_json(json_path):
    """
    Reads a JSON file from a given path and returns its content as a dictionary.
    
    Parameters:
        json_path (str): Path to the JSON file.
        
    Returns:
        dict: The content of the JSON file, or None if an error occurs.
    """
    logging.debug("Reading JSON file: " + json_path)
    normalized_path = valid_path(json_path)
    if normalized_path is None:
        logging.error("read_json: Invalid JSON path provided.")
        return None

    try:
        with open(normalized_path, 'r') as json_file:
            return json.load(json_file)
        
    except Exception as e:
        logging.error(f"read_json: Failed to read JSON file at {normalized_path}: {e}")
        return None

def write_json(json_path, json_dict, pretty=False, convert_numpy_array=False):
    """
    Saves a dictionary to a JSON file.
    
    Parameters:
        json_path (str): The path to save the JSON file.
        json_dict (dict): The dictionary to save.
        pretty (bool): Whether to format the JSON output. Defaults to False.
        convert_numpy_array (bool): Convert numpy arrays to lists before saving. Defaults to False.
    """
    logging.debug(f"Writing JSON to {json_path} with pretty={pretty}.")
    if convert_numpy_array:
        json_dict = convert_numpy_array_to_list(json_dict)
    
    try:
        with open(json_path, 'w') as json_file:
            if pretty:
                json.dump(json_dict, json_file, indent=4)
            else:
                json.dump(json_dict, json_file)

    except Exception as e:
        logging.error(f"save_json: Failed to save JSON file at {json_path}: {e}")

def convert_numpy_array_to_list(input_dict):
    """
    Recursively converts numpy arrays in a dictionary to lists.
    
    Parameters:
        input_dict (dict): The dictionary to process.
        
    Returns:
        dict: The processed dictionary with numpy arrays converted to lists.
    """
    if not isinstance(input_dict, dict):
        return input_dict
    
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            input_dict[key] = value.tolist()
        elif isinstance(value, dict):
            input_dict[key] = convert_numpy_array_to_list(value)
    
    return input_dict

def read_img(img_loc, to_type=np.float32):
    """
    Reads an image from disk.

    Parameters:
        img_loc (str): Location of the image to read.
        to_type (type): The numpy data type to convert the image to. Defaults to np.float32.

    Returns:
        numpy.ndarray: The read and processed image, or None if an error occurs.
    """
    logging.debug(f"Reading image from {img_loc} with conversion type {to_type}.")

    normalized_path = valid_path(img_loc)
    if normalized_path is None:
        logging.warning("Invalid image path provided.")
        return None

    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    
    if img.dtype != to_type and to_type == np.float32:
        img = img.astype(np.float32) / 255.

    else:
        logging.warning("Invalid image type provided.")
        return None

    return img

def write_img(img_loc, img):
    """
    Saves an image to the specified location.

    Parameters:
        img_loc (str): Location to save the image.
        img (numpy.ndarray): Image to save.
    """
    logging.debug(f"Writing image to {img_loc}.")

    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    
    # Attempt to save the image
    img_wrote = cv2.imwrite(img_loc, img)
    
    if not img_wrote:
        logging.warning(f"Failed to write image: {img_loc}")


# ============================ MAIN FUNCTION ================================ #


# Main function or other code
def main():

    args, logger = initialize_environment()
    
    # print log level according to logger
    hello()
    
    logger.critical("This is a critical message.")
    logger.error("This is an error message.")
    logger.warning("This is a warning message.")
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")

if __name__ == "__main__":
    main()
