# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.


## Software Engineering Tasks (More of suggestions really.)

- Priority 1:  Build Classifiction filters

  - Visualize Photos.

	- GUI that is able to grab images/thumbnails of model images within subdirectories and display them in a scrollable.
	  - Building it as a webpage may be useful and good experience.
	  - Able to view multiple images and continously scroll through.

	- Purpose 1: Scientists quick view of models
	  - Select range of runs of interest
	  - Select if viewing all images or view only a sample of images ( Ex. 1 out of 10 runs) 
	
	- Purpose 2: Classify Images
	  - Add Check boxes/buttons that allow user to classify image
	  - Ex.  No tidal distortions, Jumbled Mess, goood bridge/tail, etc
	  - Saves responce from user and stores classification somewhere


  - Train classification filters
	- Neural network.  
	  - Input 2 images ( or 1 image of their differences ) 
	  - Output classification
		- Ex. No tidal distortion, jumbled mess, etc  
	  - Trained based on model images with classification via GUI from above.
	
- Priority 2:  Streamline Model to Machine score process

  - Write new code to initaliaze a model directory
	- Reads
	  - Galaxy Zoo model files.
	  - Target Information. 

	- Creates 
	  - Run directory in correct organization
	  - Sub directories for particle files, model images, difference images, etc
	  - Info.txt in directory containing.

- Richa 
  - Build a general purpose filter for tidal distortions.  
	- Inputs
	  - 1st image is the final set of particles
	  - 2nd image are the initial particles shifted to the final locations.
	- Output 
	  - classify whether the final particles have any sort of tidal distortion
	- Training
	  - I (Matthew) can provide several thousands of inital and final images with labels indicating whether they have been tidally distroted or not.  
	  - Let me know what format you want these images in order to train the magic black box. 
	
	  
## Pete
- Statistics and graphs!
  - Read through directories gathering score files
	- Save results so they are not read everytime
	- Add command line argument for going back out and reading everything
  - New graphs
	- Comparing different galaxy pairs
	- comparing different comparison methods
	- Come up with hard split classification
	- Come up with correlation statistics
  - Gather images with same human score and different tiered machine scores


## Matt's To-Do

- Meeting Q's
  - What should I give Pete vs the students

- Preperation
  - Create a program that prepares the directory with all needed information in the info.txt file
  - Reads Galaxy Zoo files...
	- Models w/ scores
	- Target information
  - Creates
	- Directory with proper name/parents
	- Info.txt
	  - Sdss #
	  - run #
	  - generation ?
	  - target image location? 
	  - model_data
	  - human_score and wins/total
	  - galaxy brightnesses
	  - unique_name ? 
	- Get Target Image info
	  - Location
	  - center pixel loations
	  - brightness differences
	
	- Get Image Vis Info? 

- SPAM
  - Add humanScore to info.txt for all files
  - remove # of particles from info.txt
  - add # particles in particle file name

- Image Creator v2
  - Add version control to iamge creator and param visualization
  - Delete unzipped file
  - Add ability to check if image was already created with said parameter
  - Discuss with Dr. Wallin how to check brightness ratios between one galaxy and the other.

- Comparison methods
  - Start by filtering out those that don't change much
  - Make 1 method name variable instead of boolean for each method
  - Needs general improvement
  - Signal Correlation? ( Dr. Robertson ) 

- Image_Refinement
  - Manually find a good image param

- Score Analysis
  - Create Human vs Model plots for all models
  - compare different galaxies
  - extract images of models with same human score but different machine score



## Future Things To-Do

- Add instructions/readme for everything.
  - SPAM
  - image creator
  - Image refinement
  - comparison methods

- SPAM
  - Consider storing points in native c/c++ binary floating point data
  - Rename particle files
	- add 'pts.txt' to end of particles instead of '.txt'
	- better seperate initial and final points

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


- Param_Finder
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

  - Take score of all comparison methods and send through machine learning.

- Target Images
  - Not all target info files are there
	- Manually get from site?
	- Did jackson make a script to auto get images from site?
	- Do they need to be 'calibrated' like the others?


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
    - Get all target images and info files?
    - Read targets_done.txt in targets folder
      - Convert Jackson's script in 'targets' folder? 
        



