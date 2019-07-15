# Notes for galaxyJSPAM


These are notes meant for myself, Matthew Ogden, to keep and maintain but may be good for others to view. test.

## Matt's Immediate To-Do
Summer

- Make simulation data for everything
  - 100,000  ( in-prog on babbage )
	- Catostrophic failure on babbage.  Use particle created so far and continue

- Get Image-Creation going
  - Making main image creator more versitile
  - 

- Create Versitile Direct pixel comparison
  - 


- Masking
- Score add
  - resultion
  - date
  - experiment #
  - Experiment Comments

- Image creator
  - add pixel center 


- Get a base working version of everything and write brief instructions and summary in README.md. 
  - Installation w/ instructions on README.md ( Completed. )
  - jspamcli.py and basic_run ( In progress.  Partially complete )
  - Image creator ( Completed )
  - Difference code
    - Make super simple program that take two images and output file and does direct comparison.
    - creates output folder with misc info like difference method name, no other info.
    - Feed that program name into python script to create mass difference maker.
  - Image parameter finder


## Maintenance Items

- There are is a set of imgCreator files and imgClass files in img_creator and difference_code.  Are they outdated versions of the same class files? 

- Convert C++ image creation script to Makefile

- Occasional error about -bm in jspamcli.py reading as a float instead of integer for cores.
- Change jspmacli.py to just point to batch_run_files/batch_file.txt instead of typing full name. 

- Add standard daily Instructions to follow in README.md for everything.
  - jspamcli.py
  - basic_run
  - image creator
  - difference code
  - parameter finder

- JSPAMCLI.py
  - jpsamcli -i download input file broken 
  - *****batch file ALL is broken for everything except -bm*****
  - Update/combine scripts for different arguments
  - Why is 'q  10000 some# some#' being printed out?!
  - only -bm working when not using 'ALL'

- Not all target info files are there
  - Manually get from site?
  - Did jackson make a script to auto get images from site?
  - program way to get from website?
  - Do they need to be 'calibrated' like the others?

- Folder Restructure
  - Direct Difference Code has been moved to new folder

- pip3 install imageio?
- If you start at run '0' in batchfile, it will make a run '-001'? 


## Pete

  - galaxyJSPAM
    - install and begin creating some particle files.
    - Learn how to do a basic run of JSPAM.
    - Use jspamcli.py to do batch basic runs. 

  - Knowledge
    - Review papers. (Thesis, Toomre & Toomre, JSPAM, Galaxy Zoo )
    - Get an understanding what main programs do.
    - Ask Wallin what are good things to know.
    - learn what Graham is doing and how it'll relate.

  - Possible scripts you can write at some point. 
    - Create user friendly script to create model values to read into JSPAM. 
    - Get all target images and info files? ( See above in 'maintenance items' )
    - Create script that will create a run batch file. 
      - See 'batch_run_files' for sample of batch file

      - Read targets_done.txt in targets folder
      - Ask user or read file for below in batch file
        - number of particles per galaxy
        - all or some runs 
      - Convert Jackson's script in 'targets' folder? 
        



## Future Things To-Do

- Read papers by Lior Shamir
  - automatic pattern recognition
  - wndchrm 

- Rework run folder organization  ( in-prog )
  - How essential is it to keep sdss# and run# in particle file name?
  - add 'pts.txt' to end instead of '.txt'
  - change '.' to 'underscore' for clarity in jspamcli.py
  - info file assumed #particles is set once created
  - consider adding paramFile name next to image name in info file instead of relying on image name.

- Consider using native c/c++ binary floating point data maybe

- Consider making a file listing directories the other programs can read to know where output directories, source files, exe files, etc are ( in-prog )

- Modify to run on cluster.
  - jspam  ( done )
  - image creation ( in-prog )
  - difference code
  - image parameter

- Optimize how to find Image Parameters
  - Requires working image creator and comparison code.
  
- Standard image format?
  - dedicated galaxy centers and resolution in image.
  - Analyze Histogram and Adjust as new image parameter?

- New Comparison methods

  - Use all comparison methods and weight their results. 

  - Consider greater weights for pixel difference based on distance from galaxy center
  - Use inital points to create a weighted image for final image comparison
  - Feature Extraction via OpenCV
    1. Histogram of oriented gradients
    2. Scale-invariant feature transform
  - Pattern recognition software
    - SVD facial recognition
  - Machine Learning/ Tensor Flow
  - WNDCHRM - shamir


