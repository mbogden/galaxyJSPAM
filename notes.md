# Notes for galaxyJSPAM
These are notes meant for my own (Matthew Ogden) purposes during creation and organizing code.  May be good for others to view.


## Matt's Immediate To-Do
Summer

- Make simulation data for everything
  - 100,000  ( in-prog on babbage )
	- Catostrophic failure on babbage.  Use particle created so far and continue
	- Forgot I had things running on 

- Spam Data
  - updates to zooRun.py
	- add zoo model data to notes.txt in run directories
	- add humanscore to into.txt

- Image Creator v2
  - add option to delete unziped file

- Comparison methods
  - Create Versitile Direct pixel comparison
	- Use inital particle file to create a 'mask' of pixels to ignore
	- Use distance from center of galaxy as a weight of importance.
  
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

- Add standard daily Instructions to follow in README.md for everything.
  - basic_run
  - image creator
  - difference code
  - parameter finder


- Not all target info files are there
  - Manually get from site?
  - Did jackson make a script to auto get images from site?
  - program way to get from website?
  - Do they need to be 'calibrated' like the others?


- Folder Restructure
  - Should I add a generation folder between sdss and run directories?  I think i should

- SPAM
  - zooRun.py
	- Need to also read zoo files and write in model information and humanscore to info.txt

- Image Creator
  - Image_creator currently normalizes images.  This changes how "bright" a particle is from run to run depending on how many clump in one location.
	- Would like to change

- Score file should include
  - date
  - experiment #
  - Experiment Comments

- Useful_Bin
  - batch_execution.py
	- Noticed a core will see an empty queue and stop even though queue isn't empty. 

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
  - add 'pts.txt' to end instead of '.txt'

- SPAM
  - Consider using native c/c++ binary floating point data maybe

- Consider making a file listing directories the other programs can read to know where output directories, source files, exe files, etc are..?  ( Not likely )

- Modify to run on cluster. ( Might have accomplished this with master/worker.py
  - jspam  ( done )
  - image creation ( in-prog )
  - difference code
  - image parameter

- Optimize how to find Image Parameters
  - Requires working image creator and comparison code.
  
- Standard image format?
  - dedicated galaxy centers and resolution in image. ( Done in comparison )
  - Analyze Histogram and Adjust as new image parameter?

- New Comparison methods
  - Use all comparison methods and weight their results via machine learning!
  - Consider greater weights for pixel difference based on distance from galaxy center
  - Use inital points to create a weighted image for final image comparison
  - Feature Extraction via OpenCV
    1. Histogram of oriented gradients
    2. Scale-invariant feature transform
  - Pattern recognition software
    - SVD facial recognition
  - Machine Learning/ Tensor Flow
  - WNDCHRM - shamir


