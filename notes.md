# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.

IRB course? 
- Do not preserve any information that can identify the user.


## Score Analysis
- Statistics and graphs!
  - Stop storing images.  (Should be as simple as commenting out 1 line)
  - Save all the scores in a single csv file.
	- Change code to either read from this big file or a directory containing sdss dir

  - New graphs
	- comparing different comparison methods
	- Comparing same comparison method between different galaxy pairs
	- Come up with correlation statistics

  - Gather images with same human score and different tiered machine scores
	- Save them in unique folder and write file explaining what comparison, sdss, etc


## Matt's To-Do

- Auto create images for all 62 pairs.
  - Initial target image parameter only

Weighted Masks

- Optimize a random weighted mask for human vs machine score correlation.
  - pixel diff
  - correlation
  - Binary correlation

- Analyze image, where is the highly weighted regions?
  - radial distance from galaxy?
  - Edges of galaxy?
  - Random? Dependent on galaxy pair?  
- Can we recreate this mask? 
  - Analyze it wiht 1614, recreate with results and apply to rest.

- Bin
  - Bin image in 10x10, etc. And apply weighted mask to them, then average together

- Comparison methods
  -MachiceMethod class
	- Seperate function will return difference images
	- Seperate function for writing/appending score to csv file
	  - will need to pass run directory


## Future Things To-Do

- Add instructions/readme for everything.
  - Simulator
  - image creator
  - Image refinement
  - comparison methods

- General purpose simulated annealing program.
  - Build special modules that contain
	- cost function
	- List of things to change
	- list of size for changes
	- plot function

- Perturbedness score
  - Consider making both pure white before comparing

- Neural network to build weight based on weights for existing 62 pairs? 

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
		- simulated annealing for pixel difference
	  - Different weighted sections of image 
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

