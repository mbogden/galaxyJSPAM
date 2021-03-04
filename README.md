# galaxyJSPAM

## Table of Contents
- [Getting Started](#gettingstarted)
    - [Installation](#installation)
- [Instructions](#Instructions)
- [Overview](#overview)
- [References](#references)


## Getting Started<a id="gettingstarted">

### Installing<a id="installation">

Currently, active work is being done in a student's repository on GitHub and can be downloaded via git command.  Move into the directory and run the Makefile to install. 
```
git clone https://github.com/mbogden/galaxyJSPAM.git
cd galaxyJSPAM
make
```

If you are making modifications, checkout a different branch
- For Software Engineering execute `git checkout working_SE` 

####Python install
I am currently testing virtual environments with this package.  To initialize please run the following. This should create a new virtual environment, and install needed python packages.

```
python3 -m venv env
source env/bin/activate
python3 -m pip --upgrade pip
python3 -m pip install -r requirements.txt
```

Once you have installed the required packages, you will need to activate the environment whenever you want to run SIMR.  

## Instructions<a id="Instructions">
    
    
### Activate environment
    
```
source env/vin/activate
```
    
### Python Notebook
    
There are several notebooks in the Notebook folder (Ex. Template.ipynb) that show how to import and use several of functions located in this suite.


### Terminal Instructions
The primary program is `python3 main_simr.py` and can be activated in a number of ways.

#### Input Arguments

The main program features many additional arguments that are optional. Note all arguments start with a single dash "-".
___
- `-printBase`   (Default: True)  Program will print basic information while operating.
- `-printAll`    (Default: False) Program will print all optional information statements found in most functions.
- `-nProc #`    (Default: -1)  Operations that can be parallel, will run using n # of processers.  Note: -1 uses all available cores minus one. 
___
- `-runDir path/to/run/directory/`   Path to a model directory from a target into the program.
- `-targetDir path/to/target/directory/`   Path to a target directory into the program.
- `-dataDir path/to/directory/`    Path to directory that contains many target directories.
___
- `-paramLoc path/to/score/param.json`    Path to an existing score parameter file.
- `-targetLoc path/to/score/target.png`    Path to an existing target image of colliding galaxies.
___
- `-newScore`  (Default: False) Tells program to create a new machine score.
- `-newImg`    (Default: False) Tells program to create a new model image. (TODO)
- `-newSim`    (Default: False) Tells program to create a new simulation of a model. (TODO)
- `-newAll`    (Default: False) Tells program to create a new simulation, image, and score.
- `-overWrite` (Default: False) Tells program to create a new simulation, image, and/or score even if that file already exists.
___
- `newInfo`   (Default: False) Remove current information file.  Create a new one from a default copy.
- `newBase`   (Default: False) Remove the default information file and generate a new one. 
- `newRunInfo`   (Default: False) Will remove the information file for all models found in a target.
- `newRunBase`   (Default: False) Will remove the default information file and generate a new one for all models found in a target.
___

#### Common Commands
Common command line executions you may perform are as follows. 
- `python3 main_simr.py -targetDir path/to/target/ -newScore -paramLoc path/to/param.json` Tell program to point at a target directory and generate new machine scores.
- `python3 main_simr.py -printAll -runDir path/to/run/ -newImg -overwrite -paramLoc path/to/new_param.json` Tell program to focus on a single model, print all progress, create a new image, and overwrite existing images of the same name.  Useful for testing new image creation.
- `python3 main_simr.py -dataDir path/to/all/targets/ -newAll -paramLoc path/to/param.json -nProc 24`  Tell program to go through many targets creating new simulations, images, and score as needed.  Use 24 processors on current machine. 
    

    
## Project Overview<a id="overview">
    
This GitHub repository was developed to maintain code for Creating and Analyzing Models of Galactic Collision Simulations.  It is currently broken down into 4 primary Subsections.

- Simulation
- Image Creation
- Machine Score
- Score Analysis
- Support Code

### Simulation

As of now, all this software suite is designed to work with modeling data gathered during the citizen scientist project, Galaxy Zoo: Mergers.  [https://data.galaxyzoo.org/mergers.html](https://data.galaxyzoo.org/mergers.html).
    
This project uses the simulation known as JSPAM written by Dr. Wallin and Holincheck.  It takes model data and simulates a galactic collision.  Once completed, it creates particle files of the model.  These are automatically archived and stored in a zip file. 

### Image_Creator<a id="image_creator">
The image creator takes the particle files created by the JSPAM simulator and creates an image out of them.  This image is modified to try and recreate a realistic looking image to compare to target galactic images. 

### Machine_Score<a id="comparison_code">
The machine score code takes the created model images and target image and compares them.
    
### Score Analysis<a id="score_analysis">
This is primarily for analysis how affective various machine scoring methods behave.  The goal it to develop a process from simulation model to image to machine score that correlates with the human fitness scores from Galaxy Zoo Mergers. 
    
### Support Code<a id="score_analysis">
Support code features two programs as of now.
- `general_module.py`: Contains two classes and a handful of useful functions.  
    - Parallel processing class that exectutes a single given function with a queue of different argements.
    - Input Argument Class.  This class is often used when passing input arguments and other program data back and forth between modules. 
    - Misc functions.  Such as reading images, json files, validating and getting full path to files, and print statements. 
- `info_module.py`: Contains several classes that server as the interface between programs and disk storage.  Organizes the target/model directories and reads/saves files needed by the programs.  (Ex. Targets, models, and score parameters)
   
    
## References<a id="references">
A. Holincheck. *A Pipeline for Constructing a Catalog of Multi-method Models
of Interacting Galaxies*. PhD thesis, George Mason University, 2013.

A. J. Holincheck, J. F. Wallin, K. Borne, L. Fortson, C. Lintott, A. M. Smith, S. Bamford, W. C. Keel, and M. Parrish. Galaxy Zoo: Mergers - Dynamical models of interacting galaxies. , 459:720–745, June 2016. doi: 10.1093/mnras/stw649.

J. F. Wallin, A. J. Holincheck, and A. Harvey. JSPAM: A restricted three-body code for simulating interacting galaxies. *Astronomy and Computing*, 16:26–33, July 2016. doi: 10.1016/j.ascom.2016.03.005.