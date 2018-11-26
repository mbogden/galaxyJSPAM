# WallinCode

## Table of Contents
- [Overview](#overview)
- [Get Started](#getstarted)
- [Targets Directory](#targets)
- [jspamcli.py](#jspamcli.py)
- [data_tools](#data_tools)
- [image_creator](#image_creator)
- [comparison_code](#comparison_code)

## Overview<a id="overview">
This is the development branch of a fork of
[Wallincode](https://github.com/JSPAM-Manga/WallinCode).

The recent developments in this fork of JSPAM-Manga focus on creating a machine
scoring mechanism for comparing
rendered galaxy merger models to the morphology of their real counterparts.
These machine scores can then be used in conjunction with citizen science
efforts to score these same models, effectively reducing the human interaction
time needed to determine whether a particular model is "good" or "bad." Further,
the availability of a machine scoring mechanism will open the door to
incorporation of various machine learning algorithms to immediately recognize
and remove obviously "bad" models,
thereby removing the need for a human-in-the-loop altogether.

## Get Started<a id="getstarted">
**MAKE SURE TO DO ALL WORK INSIDE OF THE VIRTUALENVIRONMENT**

To simplify things, we have included the entire `virtualenv-15.2.0` package
found at
[https://pypi.python.org/pypi/virtualenv/15.2.0](https://pypi.python.org/pypi/virtualenv/15.2.0). To setup
the virtual environment install the required packages, issue the following
commands from the root directory:

```
python3 virtualenv-15.2.0/virtualenv.py -p python3 ./env
```

At this point, the virtualenv can be activated by entering

```
source env/bin/activate
```

To install the required packages, issue the following command:

```
pip install -r requirements.txt
```

## Targets Directory<a id="targets">
Real images (both uncalibrated and calibrated from SDSS DR7) and disk
information for each target can be found in `targets`. The calibrated images
are created using 
Real images of the target galaxies are contained in the `target_images`
directory. These are to be used as the reference images for comparison in
testing.

To my knowledge, the original dataset is based off of images from SDSS DR7, but
the image directory will contain processed images of those same targets using
data from SDSS DR14.

## jspamcli.py<a id="jspamcli.py">
This fork of WallinCode contains `jspamcli.py`, a python3 script that runs
specific runs or specific ranges of runs from the overlapping galaxy pairs
table at
[https://data.galaxyzoo.org/mergers.html](https://data.galaxyzoo.org/mergers.html).
All target input files have been provided in the `input` directory.

**NOTE:** The input files in the `input` directory contain only the run
information for runs that recieved a human score in the Galaxy Zoo: Mergers
project. There are 66,395 total runs in all of the input files combined.

```
jspamcli accepts the following command line options:

    -i  : run interactively
    -bi : batch process (interactively...)
    -b  : batch process
    -bm : batch process on multiple cores
    -g  : GIF Creation Tool

```

**WORKING ON ADDING USAGE INFORMATION...**

## data_tools<a id="data_tools">
This is a package that will contain any data tools that can be written as a
general purpose tool. Right now, it contains
- `structure.py`: This is a general_purpose directory structure creator.
- `get_target_data.py`: This is a module that scrapes the mergers.html page
for links to the zipped target data files. This may not belong in the package,
but as of right now that is its home.

## image_creator<a id="image_creator">

## comparison_code<a id="comparison_code">

# Notes
All development is currently being done on the
[development](https://github.com/jacksonlanecole/WallinCode/tree/development)
branch of this fork. Check it out there!

# References
A. Holincheck. *A Pipeline for Constructing a Catalog of Multi-method Models
of Interacting Galaxies*. PhD thesis, George Mason University, 2013.

A. J. Holincheck, J. F. Wallin, K. Borne, L. Fortson, C. Lintott, A. M. Smith, S. Bamford, W. C. Keel, and M. Parrish. Galaxy Zoo: Mergers - Dynamical models of interacting galaxies. , 459:720–745, June 2016. doi: 10.1093/mnras/stw649.

J. F. Wallin, A. J. Holincheck, and A. Harvey. JSPAM: A restricted three-body code for simulating interacting galaxies. *Astronomy and Computing*, 16:26–33, July 2016. doi: 10.1016/j.ascom.2016.03.005.
