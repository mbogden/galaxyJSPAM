# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
May be good for others to view.

IRB course? 
- To not preserve any information that can identify the user.

# Matt's Start up To-Do
- (When do-able) Read Me files? 
- (Working) Virtual Environment?
- (Working) Cluster to use?
- (Later) wndchrm
- (Now) Pipeline
- (Later) Have 100 models from each target on my machine
- Create Target status report?

## Big picture
- Create automatic pipeline for evolution
	- Get function to create status update!
	- Create reports for any missing items. 
	- Implement NN for each target

- Simulator
	- Produce initial impact approximation w/o points
	- Stop if impact is too low.

- Image Creator
	- Rotating galaxies still not working
	- Do Total variation denoising on Model images
	- Params, clause if arguments are incomplete to create image.

- Image Comparison.
	- (HIGH) Rewrite how params are passed to machine score method functions
	- Implement new low hanging machine score
	- Reimplement WNDCHRM
	- Weighted Mask
	- ? Seperate function will return difference images

- Score Analysis
	- Incorporate score analysis to be based on new score parameters

## Now...
- Create single layer NN for all targets
- Create pts, imgs, scores for ALL

### Future

- Analyze Weighted NN image, where is the highly weighted regions?
  - Can we recreate this mask? 
  - radial distance from galaxy centers?
  - tails and bridges

  - New Methods
	- Zernike moments
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
	- Use wandering mask to view brightness base on radial coordinate system on galaxies. 
	  - characterize with functions or hard point and lines. 
	  - Use line to determine direction of brightness.
	- Weigh all machine scores together. 
	  - Take score of all comparison methods and send through machine learning.


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

