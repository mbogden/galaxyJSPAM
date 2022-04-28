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

#### Python package install
I am currently testing virtual environments with this package.  To initialize please run the following. This should create a new virtual environment name "simr_env", and install needed python packages. NOTE: Then name "simr_env" is arbitrary and you can name it however you please.

```
python3 -m venv simr_env
source simr_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

##### WNDCHARM Install
Much of the Image Feature Extraction used in this Project utilizes the open-source software WNDCHRM. [Pub][1] [Repo][2].
    
    
The following is my, Matthew Ogden's, notes on installing WNDCHRM for the first time on a personal machine. I cannot guarantee they will work in all scenarios.
    
``` 
    sudo apt-get install build-essential 
    sudo apt-get install -y libtiff-dev 
    sudo apt-get install libfftw3-dev 
    
    git clone https://github.com/wnd-charm/wnd-charm.git 
    cd wnd-charm 
    sudo ./build.sh 
    
``` 

## Instructions<a id="Instructions"> 
    
### Activate environment 
    
    Since the python packages were installed into a virtual environment, you must launch the virtual environment whenever you run code.
     
```
source simr_env/bin/activate 
```
    
### Python Notebook 
    
There are several notebooks in the Notebook folder (Ex. Template.ipynb) that show how to import and use several functions located in this suite. 

### Terminal Instructions 

The primary program can be launched in two different methods! 
    - `python3 main_simr.py` 
    - `mpirun -n 4 python3 main_simr.py` 

#### Input Arguments

The main program allows for several arguments. Note all arguments start with a single dash "-".  
    - For arguments without a following value, the default value is set to a booleon "True".  You can type a "true" or "FALSE" (Case-insensitive) to set the bool value to True or False. 
    - All arguments with a following value are assumed to be a string. 

___
- `-printBase`   (Default: True)  Program will print basic information while executing.
- `-printAll`    (Default: False) Program will print all optional information statements found throughout the code.  
- `-printBaseRun`(Default: False) Print base model info while iterating through target models.
- `-printAllRun` (Default: False) Print all optional model info while iterating through target models.
___
- `-runDir path/to/run/directory/`         Path to a model directory into the program.
- `-targetDir path/to/target/directory/`   Path to a target directory into the program.
- `-dataDir path/to/directory/`            Path to directory that contains many target directories.
___
- `-scoreParamLoc path/to/score/score_param.json`     Path to an existing score parameter file.
- `-scoreParamName name_of_score_aram`               Name of a score parameters file already saved in target's score_parameters folder.
- `-targetLoc path/to/score/target.png`    Path to an existing target image of colliding galaxies.
___
- `-newGen`  (Default: False)    Used for testing new MPI Queue for creating new models
- `-gaExp`   (Default: False)    Short for "Genetic Algorithm Experiment".  Directs program to begin a genetic evolution of models over target given.
- `-gaLocName`   (Default: target)   Possible locations Workers use when generating model data to create a score.  
- `-gaParamLoc path/to/ga_param.json`   Location for genetic algorithm parameter file 
___
- `-newSim`    (Default: False) Tells program to create a new simulation of a model.
- `-zipSim`    (Default: False) Tells program to zip particles files created by simulator.  Warning, files can be large.
- `-newImage`  (Default: False) Tells program to create a new model image.
- `-newFeats`  (Default: False) Tells program to create WNDCHRM features out of model image.
- `-newScore`  (Default: False) Tells program to create a new machine score.
- `-newPlot`   (Default: False) Tells program to create series of plots for a target.
- `-newAll`    (Default: False) Tells program to create a new simulation, image, and score.
- `-overWrite` (Default: False) Tells program to create a new files even if they already exist.
___
- `-newInfo`    (Default: False) Remove current information file.  Create a new one from a copy of the base.
- `-newBase`    (Default: False) Remove the base information file and generate a new one. 
- `-newRunInfo` (Default: False) Will remove the information file for all models found in a target.
- `-newRunBase` (Default: False) Will remove the base information file and generate a new one for all models found in a target.
___
- `-startRun`  (Default: 0)  While iterating through target models, start with N'th model
- `-endRun`    (Default: -1) While iterating through target models, end with N'th model. (-1 go until end)
- `-skipRun`   (Default: 1)  While iterating through target models, skip to everyh N'th model
___
- `-normFeats`  (Default: False) Takes WNDCHRM features created in runs and normalizes them.  Must be paired with -normName or -normLoc.
- `-normName file_name` Loots for a feature normalization file in target's WNDCHRM directory.
- `-normLoc path/to/file.json`   Looks for feature normalization file in specified path.
- `-wndchrmAnalysis`   Performs wndchrm analysis
___

### Common Commands
    
#### Initialize a target and generate basic direct image comparison scores 
`python3 main_SIMR.py -dataDir path/to/all/targets/ -newInfo -newBase -newRunInfo -newRunBase -newScore -newImage -paramName zoo_0_direct_scores` 
    
#### Generate basic WNDCHRM images, features values, and normalize them.
`python3 main_SIMR.py -targetDir path/to/target/ -newImage -paramName chime_0 -newFeats -normFeats -normName norm_chime_0` 
    
#### Tell program to point at a target directory and generate new machine scores from a parameter file you've created.
`python3 main_SIMR.py -targetDir path/to/target/ -newScore -paramLoc path/to/param.json` 
    
#### Tell program to focus on a single model, create a new image, overwrite existing images and print all progress.  Useful for testing new image creation.
`python3 main_SIMR.py -printAll -runDir path/to/run/ -newImage -overWrite -paramLoc path/to/new_param.json` 
    
#### Tell program to go through many targets creating new simulations, images, and score as needed.  Use 24 processors on current machine. 
`mpirun -n 24 python3 main_SIMR.py -dataDir path/to/all/targets/ -newAll -paramLoc path/to/param.json`  
    
    
## Project Overview<a id="overview">
    
This GitHub repository was developed to maintain code for Creating and Analyzing Models of Galactic Collision Simulations.  It is currently broken down into 5 primary Subsections.

- Simulation
- Image Creation
- Machine Score
- Score Analysis
- Support Code

### Simulation

As of now, this software suite is designed to work with modeling data gathered during the citizen scientist project, Galaxy Zoo: Mergers.  [https://data.galaxyzoo.org/mergers.html](https://data.galaxyzoo.org/mergers.html).
    
This project uses the simulation known as JSPAM written by Dr. Wallin and Holincheck.
It takes model data and simulates a galactic collision.
Once completed, it creates particle files of the model.
These are automatically archived and stored in a zip file.

### Image_Creator<a id="image_creator">
The image creator takes the particle files created by the JSPAM simulator and creates an image out of them.
This image is modified to try and recreate a realistic looking image to compare to target galactic images. 

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
   
## Score Parameters (TODO)
    For each model, there are a variety of methods for scoring how well the model fits observation data.  The primary goal of this software suite is to find a fitness scoring method that matches.  
    Our primary method for getting this fitness score is by simulating the orbital parameters of the model in SPAM.  The simulation creates orbiting stars that get tidally displaced as the galaxies interact. 
    
    
### NOTE:  All code is assuming you have a group of possible score parameters. 

### Example JSON
```json
'zoo_2': {'cmpArg': {'cmpMethod': 'correlation'},
           'imgArg': {'blur': {'size': 5,
                               'type': 'gaussian_blur',
                               'weight': 0.75},
                      'comment': 'Smaller '
                                 'Image '
                                 'by '
                                 'Matthew.',
                      'galaxy_centers': {'px': 50,
                                         'py': 50,
                                         'sx': 100,
                                         'sy': 50},
                      'image_size': {'height': 100,
                                     'width': 150},
                      'name': 'zoo_2',
                      'normalization': {'norm_constant': 2.5,
                                        'type': 'type1'},
                      'radial_const': [-1.5,
                                       -1.5],
                      'target_id': '587722984435351614'},
           'name': 'zoo_2',
           'scoreType': 'target',
           'simArg': {'nPts': '100k',
                      'name': '100k'},
           'targetName': 'zoo_2'}
```
    
## References<a id="references">
A. Holincheck. *A Pipeline for Constructing a Catalog of Multi-method Models
of Interacting Galaxies*. PhD thesis, George Mason University, 2013.

A. J. Holincheck, J. F. Wallin, K. Borne, L. Fortson, C. Lintott, A. M. Smith, S. Bamford, W. C. Keel, and M. Parrish. Galaxy Zoo: Mergers - Dynamical models of interacting galaxies. , 459:720–745, June 2016. doi: 10.1093/mnras/stw649.

J. F. Wallin, A. J. Holincheck, and A. Harvey. JSPAM: A restricted three-body code for simulating interacting galaxies. *Astronomy and Computing*, 16:26–33, July 2016. doi: 10.1016/j.ascom.2016.03.005.

[1]: https://scfbm.biomedcentral.com/articles/10.1186/1751-0473-3-13 "WNDCHARM Publication"
[2]: https://github.com/wnd-charm/wnd-charm "WNDCHRM GitHub Repository"