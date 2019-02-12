# Notes for galaxyJSPAM


These are notes meant for myself, Matthew Ogden, to keep and maintain but may be good for others to view. test.

## Immediate To-Do

- Get a base working version of everything
  - Installation. ( Completed. Do 'maintenance items' below. )
  - jspamcli.py and basic_run ( Working. Partially complete )
  - Image creator
  - Difference code
  - Image parameter finder

## Pete
  - GitHub
    - Make a github account
    - Github instruction sheet
    - Give access to write to galaxyJSPAM github repository.

  - Download galaxyJSPAM
    - install and begin creating some particle files.

  - Knowledge
    - Review papers
    - Get an understanding what main programs do.
    - Ask Wallin what are good things to know.
    - learn what Graham is doing and how it'll relate.

  - Possible things to do.
    - Get all target images and info files? ( See below in 'maintenance items' )
    - Create script that will create a run batch file. 
      - See 'batch_run_files' for sample of batch file
      - Convert Jackson's script in 'targets' folder? 
      - Read targets_done.txt in targets folder
      - Ask user or read file for below in batch file
        - number of particles per galaxy
        - all or some runs? 
    
    - Create user friendly script to create model variables to read into JSPAM. 



## Maintenance Items
- Installation
  - Add git clone command in readme.md
  - virtualenv was moved to archive folder.  No longer needed? Move back? 
  - Version number for 'futures' is too new?
  - add 'pip install lxml' (To requirements?)

- Separate installation and daily startup instructions for jspamcli.py in readme.

- Add how you run jspamcli.py in command line to ReadMe with different options.
  - Consider adding execution instructions for all code.

- JSPAMCLI.py
  - jpsamcli -i download input file broken 
  - *****batch file ALL is broken for everything except -bm*****
  - Update/combine scripts for different arguments
  - Why is 'q  10000 some# some#' being printed out?!

- Not all target info files are there
  - Manually get from site?
  - Did jackson make a script to auto get images from site?
  - program way to get from website?
  - Do they need to be 'calibrated' like the others?



## Future Things To-Do

- Optimize how to find Image Parameters
  - Requires working image creator and comparison code.
  
- Standard image format?
  - dedicated galaxy centers and resolution in image.

- New Comparison methods
  - Consider greater weights for pixel difference based on distance from galaxy center
  - Feature Extraction via OpenCV
    1. Histogram of oriented gradients
    2. Scale-invariant feature transform
  - Pattern recognition software
    - SVD facial recognition
  - Machine Learning/ Tensor Flow
  - Use all comparison methods and weight their results. 
- Modify to run on cluster.
- Analyze Histogram and Adjust as new image parameter?
