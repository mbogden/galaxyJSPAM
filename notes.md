# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.

IRB course? 
- Do not preserve any information that can identify the user.


## Software Engineering Tasks (More of suggestions really.)
- This contains quick notes to be added to needed files later



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

- Image Creator v2
  - Add version control to image creator and param visualization
  - Alter brightness of galaxies seperately once luminosity is added to info.txt 
  - Have seperate radial constants for each galaxy 
  - (DONE) Add ability to create model image from inital particle file.
  - (DONE) Delete unzipped file

- Comparison methods
  - Check that new target info files are actual target centers
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

