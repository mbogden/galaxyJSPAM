# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.

IRB course? 
- Do not preserve any information that can identify the user.


## Software Engineering Tasks (More of suggestions really.)
- This contains quick notes to be added to needed files later



<<<<<<< HEAD
=======
- Streamline Model to Machine score process
  - Write new code to initaliaze a model directory
	- Reads and processes
	  - Zoo models @ Input_Data/zoo_models/from_site/*SDSS*.txt
	  - target images and select files @ Input_Data/target_images/*SDSS*/
	  - Image parameter file @ Input_Data/image_parameters/
	- Creates
	  - Directory Structure
		- mainDir (input var) -> sdssDir (#) -> genDir (0) -> runDir (line # in file)
	  - Extra Folders in runDir
		- particle_files
		- model_images
		- misc_images
	  - info.txt
		- you can find contents for this file in info_template.txt
		- If any info is missing, just have a blank spot for that location in info.txt

  - Modify simulator_v2.py.
	- I have seperate notes on this later.


- Visualize Photos.
  - Beginning:
	- Create PHP file.  
	- Displays all images found in a directory.
	  - Add button to specificy directory of interest? 
	- Format page to scroll through images.
	- Point to sdss directory and display model images found in subdirectories.
	  - directory structure
	- Display select info from info.txt files in subdirectories with image
	  - sdss_name, gen#, run#, human_score

  - Purpose 1: Scientist's quick view of models
	- Select range of runs of interest
	- Select if viewing all images or view only a sample of images ( Ex. 1 out of 10 runs) 
	- Display image run # and humanscore
  
  - Purpose 2: To classify Images
	- Add Check boxes/buttons that allow user to classify image
	- Ex.  No tidal distortions, Jumbled Mess, goood bridge/tail, etc
	- Save response from user and store classification in info.txt


- Train classification filters
  - Tidal Distortion filter  

	- Input 
	  1. model image 
	  2. inital image

	- Output classification
	  - "good" or "bad" tidal distortion

	- Training process
	  - First training set can be found at '/nsfhome/mbo2d/Public/training_image_set_1.zip'.  Upzip
		- Will create more and better training sets later.
	  - Two directories.
		- goodDir: Contains image pairs for "good" tidal distortions
		- badDir: Contains image pairs for "bad" tidal distortions
	  - Image pair format: images with same sdss name and run number are a pair
		- model image: sdssName_runNumber_model.png
		- init image : sdssName_runNumber_init.png

  - Jumpled Mess Filter
	- Very similar to above
	- Read in two images and identify if they are "too jumbled"
>>>>>>> cc42b0a4d9c1f6f8ce6c1c1ba57fc112a7106478
## Pete
- Statistics and graphs!
  - Read through directories gathering score files
	- Save results so they are not read everytime
	- Add command line argument for going back out and reading everything
  - New graphs
	- Comparing different galaxy pairs
	- comparing different comparison methods
	- Come up with correlation statistics
  - Gather images with same human score and different tiered machine scores


## Matt's To-Do

- Write program to add human_score to info.txt in all 66,000 models on babbage

<<<<<<< HEAD
- Put target images, info files, and .pair files in target folders

- SPAM
  - remove # of particles from info.txt
  - add # particles in particle file name
=======
- Modify simulator_v2.py
  - Primary Cmd Line Arguments
	- Run directory
	- # of particles for each galaxy
  - reads info.txt to get all needed info from info.txt
  - Change working directory to particle_file directory before calling SPAM code
  - Rename particles as numParticles_pts.000 or numParticles_pts.101 ( Ex. 1000_pts.000 )
  - zip them as numParticles_000.zip (Ex. 10000_000.zip, 200_101.zip )

 - add # particles in particle file name
>>>>>>> cc42b0a4d9c1f6f8ce6c1c1ba57fc112a7106478

- Image Creator v2
  - Add version control to image creator and param visualization
  - Alter brightness of galaxies seperately once luminosity is added to info.txt 
  - Have seperate radial constants for each galaxy 
  - (DONE) Add ability to create model image from inital particle file.
  - (DONE) Delete unzipped file

- Comparison methods
  - ALL comparison methods only return a machine score!
	- Seperate function will return difference images!
	- Will they all only take 2 images though?  No.... 
  - Make 1 method name variable instead of boolean for each method?


## Future Things To-Do

- Add instructions/readme for everything.
  - Simulator
  - image creator
  - Image refinement
  - comparison methods

- SPAM
  - Consider storing points in native c/c++ binary floating point data

- Image_Creator
  - currently views galaxy points as [[x1,y1],[x2,y2]]
	- needs to be transposed

  - look into different normalization options
	- Add original normalization 
	- Possibly completing remove in favor of additional radial constant

  - Radial_Constants
	- Make seperate radial constant for both galaxies
	- Add second radial brightness constant

  - clean up code
	- make param class
	- make galaxy class

  - Add method for comparing and changing brightness between the two galaxies.


- Refine_Image
  - Finding Local Max does not work...  
  - May need to find better comparison methods first.
  - Future methods
	- RSAP
	- EigenValue max finding via grid search
 
- Add more image preperation!
  - Adjust brightnesss so mean is the same
	- Maybe mean standard deviation is same too
  - Analyze Histogram and adjust? ( Ask Laurel )

- Comparison_Methods

  - Add way to check score.csv to check if comparison on image has already been used
  - make difference between main and main function so paramfinder can import directly

  - currently views galaxy points as [[x1,y1],[x2,y2]]
	- needs to be transposed

  - Methods
	- Mask
	  - Solid mask
	  - Weighted mask
	- OpenCV feature extraction
	  - Histogram of Oriented gradients
	  - Scale-Invariant feature transform
	- Machine Learning / Tensor Flow
	- Pattern Recognition Software
	  - SVD facial recognition
	- Read papers by Lior Shamir
	  - automatic pattern recognition
	  - wndchrm 

	- Signal Correlation? ( Dr. Robertson ) 

  - Take score of all comparison methods and send through machine learning.

- Target Images
  - Not all target info files are there
	- Manually get from site?
	- Did jackson make a script to auto get images from site?
	- Do they need to be 'calibrated' like the others?



## Modify image_creator_v3.py
- Complete simulator_v2.py modification first.
- Tasks
  - read info.txt for most information
  - check for proper image_parameter_verion#. Exit if not compatible
  - Check if model image of that image_parameter exists.  Exit if already present.
  - Save model image and init image in 'model_images' folder.  All other images go to 'misc_images' folder.


## compare_v2.py
- Tasks
  - Retrieve Target information from info.txt
	- target location
	- target pixel centers
  - Check if comparison method has already been applied to model image in scores.csv file
  - (Matthew/Optional) Image preperation: Analyze brigtness of photo and adjust to match model image to target image
  - (Matthew) Redo all comparison methods to only return machine score
  - (Matthew) Create seperate function to return images from comparison method program 
  - (Matthew) Implement mask comparison with pixel differences


## Begin Building Model Pipeline Code
- Core Pipeline
  - Simulate 100k pts -> Image Creations -> Machine Score

- Filter Pipeline ( Once filter is ready to test )
  - Run before core.
  - Simulate 10k pts -> apply filter
  - If filter decides model is "good", apply basic pipeline

