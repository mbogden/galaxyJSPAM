# galaxyJSPAM

## Table of Contents
- [Overview](#overview)
- [Getting Started](#gettingstarted)
- [Specifications](#specifications)
    - [jspamcli.py](#jspamcli.py)
    - [Targets Directory](#targets)
    - [data_tools](#data_tools)
    - [image_creator](#image_creator)
    - [comparison_code](#comparison_code)
- [Notes](#notes)
- [References](#references)

## Overview<a id="overview">

This README.md is also incomplete and in-progress


## Getting Started<a id="gettingstarted">

### Installing

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
python -m pip install -r requirements.txt
```

Once you have installed the required packages, you will need to activate the environment whenever you want to run SIMR.

DAILY INSTRUCTIONS - Start up
```
source env/vin/activate
```

DAILY INSTRUCTIONS - Close
```
deactivate
```


## *Add test and daily instructions for...*
- Simulation
- Image Creation
- Machine Comparison
- Score Analysis


### Particle Files


[https://data.galaxyzoo.org/mergers.html](https://data.galaxyzoo.org/mergers.html).
All target input files have been provided in the `input` directory.


**NOTE:** The input files in the `input` directory contain only the run
information for runs that recieved a human score in the Galaxy Zoo: Mergers
project. There are 66,395 total runs in all of the input files combined.



### Targets Directory<a id="targets">
Real images (both uncalibrated and calibrated from SDSS DR7) and disk
information for each target can be found in `targets`. The calibrated images
are created using 
Real images of the target galaxies are contained in the `target_images`
directory. These are to be used as the reference images for comparison in
testing.

To my knowledge, the original dataset is based off of images from SDSS DR7, but
the image directory will contain processed images of those same targets using
data from SDSS DR14.


### data_tools<a id="data_tools">
This is a package that will contain any data tools that can be written as a
general purpose tool. Right now, it contains
- `get_target_data.py`: This is a module that scrapes the mergers.html page
for links to the zipped target data files. This may not belong in the package,
but as of right now that is its home.

### image_creator<a id="image_creator">

### comparison_code<a id="comparison_code">

## Notes<a id="notes">
All development is currently being done on the
[development](https://github.com/jacksonlanecole/WallinCode/tree/development)
branch of this fork. Check it out there!

## References<a id="references">
A. Holincheck. *A Pipeline for Constructing a Catalog of Multi-method Models
of Interacting Galaxies*. PhD thesis, George Mason University, 2013.

A. J. Holincheck, J. F. Wallin, K. Borne, L. Fortson, C. Lintott, A. M. Smith, S. Bamford, W. C. Keel, and M. Parrish. Galaxy Zoo: Mergers - Dynamical models of interacting galaxies. , 459:720–745, June 2016. doi: 10.1093/mnras/stw649.

J. F. Wallin, A. J. Holincheck, and A. Harvey. JSPAM: A restricted three-body code for simulating interacting galaxies. *Astronomy and Computing*, 16:26–33, July 2016. doi: 10.1016/j.ascom.2016.03.005.

