# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.

IRB course? 
- To not preserve any information that can identify the user.

# Matt's To-Do

## Big picture
- Create automatic pipeline
  - (DONE) Generate points.  
  - (Mostly Done) Generate Images
  - Generate machine scores.
  - Check for runs missing points/imgs/scores

## Now...
- Add instructions/readme for everything.
- Finish auto pipeline
  - Fix info module
  - Update Simulator to work with new code and info.  
- Create Auto Generated Score Analysis report 
- Create checker
  - Checks for pts, images, scores
  - Create reports for any missing items. 
- Create single layer NN for all targets
- Create pts, imgs, scores for ALL

## SIMR Pipeline
- Core Pipeline
  - Simulator -> Image Creations -> Machine Score

## Simulator
- Update simulator
- Calulate Beta value, filter if lower than 0.1

##Image Creator
### Now...
- (In prog) Create images for everything
  - Create checker to see if all runs have images

### Future
- Compare unperturbed model with target and see how it correlates with scores.
- Use np.histogram to autobin giant list of points optimially.
- Create std image param based on # of particles
  - no radial distance, just pts
- modify galaxy centers and img size
- Chnage init image to 'misc_images' folder.  
- Do Total variation denoising on Model images


## Machine scores

### Now
- Create machine score for EVERYTHING
- Create single layer NN for each target

### Future
- Seperate function will return difference images

- Analyze Weighted NN image, where is the highly weighted regions?
  - Can we recreate this mask? 
  - radial distance from galaxy centers?
  - tails and bridges

  - New Methods
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


## Info module
- Figure out why it's hesitiating to read in json file
  - May need to save list of scores in seperate csv file

## MISC Items
- General purpose 2d and 3d plot
  - Allows users to select axis values from dropdown. 
  - Clicking a point shows image and data about that run.
  - View runs from 3d? 

## Score Analysis
- New graphs
  - comparing different comparison methods
  - Comparing same comparison method between different galaxy pairs
  - Come up with correlation statistics

- Gather images with same human score and different tiered machine scores
  - Save them in unique folder and write file explaining what comparison, sdss, etc

- SVD on images and see which show most variance>? 

