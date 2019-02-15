# Notes for galaxyJSPAM


These are notes meant for myself, Matthew Ogden, to keep and maintain but may be good for others to view. test.

## Matt's Immediate To-Do

- Get a base working version of everything and write brief instructions and summary in README.md. 
  - Installation w/ instructions on README.md ( Completed. )
  - jspamcli.py and basic_run ( In progress.  Partially complete )
  - Image creator ( Completed )
  - Difference code
  - Image parameter finder


## Pete
  - GitHub
    - Give access to write to galaxyJSPAM github repository.

  - galaxyJSPAM
    - install and begin creating some particle files.
    - Learn what is takes to do a basic run of JSPAM.
    - Use jspamcli.py to do batch basic runs. 

  - Knowledge
    - Review papers. (Thesis, Toomre
    - Get an understanding what main programs do.
    - Ask Wallin what are good things to know.
    - learn what Graham is doing and how it'll relate.

  - Possible scripts you can write at some point. 
    - Create user friendly script to create model variables to read into JSPAM. 
    - Get all target images and info files? ( See below in 'maintenance items' )
    - Create script that will create a run batch file. 
      - See 'batch_run_files' for sample of batch file
      - Convert Jackson's script in 'targets' folder? 
      - Read targets_done.txt in targets folder
      - Ask user or read file for below in batch file
        - number of particles per galaxy
        - all or some runs - Machine Learning/ Tensor Flow
    - Use all comparison methods and weight their results. 




## Maintenance Items

- Include c++ compilation script in readme
  - Get a working script for compiling c++ image creation code. 


- Occasional error about -bm in jspamcli.py reading as a float instead of integer for cores.
- Change jspmacli.py to just point to batch_run_files/batch_file.txt instead of typing full name. 
- Add quick test for img.sh in readme.md




- Daily Instructions for anyone to follow.
  - jspamcli.py
  - basic_run
  - image creator
  - difference code
  - parameter finder

- Add how you run jspamcli.py in command line to ReadMe with different options.

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

- Rework run folder organization
  - How essential is it to keep sdss# and run# on particle files if they're in the correct directory? 
  - add 'pts.txt' to end instead of '.txt'
  - change '.' to 'underscore' for clarity in jspamcli.py
  - info file assumed #particles is set once created
  - consider adding paramFile name next to image name in info file instead of relying on image name.


- Consider making a file listing directories the other programs can read to know where output directories, source files, exe files, etc are

- Modify to run on cluster.

- Optimize how to find Image Parameters
  - Requires working image creator and comparison code.
  
- Standard image format?
  - dedicated galaxy centers and resolution in image.
  - Analyze Histogram and Adjust as new image parameter?

- New Comparison methods

  - Consider greater weights for pixel difference based on distance from galaxy center
  - Feature Extraction via OpenCV
    1. Histogram of oriented gradients
    2. Scale-invariant feature transform
  - Pattern recognition software
    - SVD facial recognition
 
 

