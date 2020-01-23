# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.

IRB course? 
- Do not preserve any information that can identify the user.

## Matt's To-Do

- Get targets ready for pipeline
  - gather targets all together
  - create default param file
	- modify galaxy centers and img size to best match galaxy

- Get Master and other branches merged together

- Create automatic pipeline for all 62 pairs
  - (DONE) Generate points.  
  - (Working) Generate Images
	- Found bug with rotating galaxy centers.... working
  - Generate machine scores.

- Particle files were zipped with entire directory path saved....  Fix that.

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

- Comparison methods
  -MachiceMethod class
	- Seperate function will return difference images
	- Seperate function for writing/appending score to csv file
	  - will need to pass run directory

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

- General purose 2d and 3d plot
  - Allows users to select axis values from dropdown. 
  - Clicking a point shows image and data about that run.
  - View runs from 3d? 

- Perturbedness score
  - Consider making both pure white before comparing
  - Try radial brightness correlation


- Target Images
  - Not all target info files are there
	- Manually get from site?
	- Did jackson make a script to auto get images from site?
	- Do they need to be 'calibrated' like the others?

- Image_Creator
  - currently views galaxy points as [[x1,y1],[x2,y2]]
	- needs to be transposed
  - Do Total variation denoising on Model images!
  - look into different normalization options
	- Add original normalization 
	- Possibly completing remove in favor of additional radial constant
  - Radial_Constants
	- Make seperate radial constant for both galaxies
	- Add second radial brightness constant
  - clean up code
	- make galaxy class


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

  - currently views galaxy points as [[x1,y1],[x2,y2]]
	- needs to be transposed

  - Auto bin image in 10x10, 124x124 etc.
	- Drastic difference between resolution? 
	- Easier to create weighted mask.
	- Apply score of different resolutions together? 

  - Methods
	- Mask
	  - Solid mask for inner disk.
	  - Sectioned weights of image. Radial function?
	  - Simulated annealing for pixel difference, ( or any machime method) 
	- OpenCV feature extraction
	  - Histogram of Oriented gradients
	  - Scale-Invariant feature transform
	- Machine Learning / Tensor Flow
	  - Create regression score
	  - Build general weighted mask based on existing 62 pairs.
	- Pattern Recognition Software
	  - SVD facial recognition
	- Read papers by Lior Shamir
	  - automatic pattern recognition
	  - wndchrm 
	- Extract spiral from galaxy center and do correlation
	  - Extract radial polar function from galaxy centers
	- Look at moments of the image? 
	  - Ask Wallin for clarification
	- Use wandering mask to view brightness base on radial coordinate system on galaxies. 
	  - characterize with functions or hard point and lines. 
	  - Use line to determine direction of brightness.
	- Weigh all machine scores together. 
	  - Take score of all comparison methods and send through machine learning.

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




