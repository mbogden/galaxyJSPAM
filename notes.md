# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
Might* be good for others to view.

# Matt's Working To-Do

- Capstone Project
    
    - Commands to remember
        - python3 main_SIMR.py -targetDir path/to/target -newInfo -newRunInfo -newBase -newRunBase 
        - python3 main_SIMR.py -dataDir path/to/data -newScore -newImage -paramName zoo_0_direct_scores
        
    - Get target pipeline working
        - (DONE) Updating old data to new layout
        - (DONE) Automate scores for direct machine comparisons
        
    - Get MPI working for a target
        - Copied/Shared target class object?
        - Tasker/Workers?
        
    - Get a single-layer neural network going
        - Gather and pickle images into a temp file/folder
        - Use GPU to make the creation faster
        - Generalize it to all targets.

## Big picture
- Create automatic pipeline for evolution
    - Get function to create status update!
    - Create reports for any missing items. 

- Simulator
    - Produce initial impact approximation w/o points
    - Stop if impact is too low.

- Image Creator
    - adjusting target image not working for zoo_2?
    - Do Total variation denoising on Model images

- Image Comparison.
    - Reimplement WNDCHRM
    - Implement Neural Networks
        - Create Single Layer Neural Network for all.
    - Weighted Mask
    - ? Seperate function will return difference images

- Score Analysis
    - Get all systems running on Roughshod.


### Future
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
    - Use wandering mask to view brightness base on radial coordinate system on galaxies. 
      - characterize with functions or hard point and lines. 
      - Use line to determine direction of brightness.
    - Weigh all machine scores together. 
      - Take score of all comparison methods and send through machine learning.

- Triple image comparison
	- Compare all Model, Unperturbated, and target image
	- Where Model and unperturbed closely match, lower weight on model image

- Masked comparison
    - Apply mask to both images 
    - Send to specified compr method 
    - Mask Creation 
        - Arbitrary ovals that capture edge of target? 
        - single-layer nn? 
        - HOG abs-val of gradient? 
  
- Sparce Correlation 
    - Take 3+ "Rings" of the image 
    - Transorm into radial coordinates? 
    - Take correlation with flat matrix 
    - Send new matrix to std comparison methods? 


## MISC Items
- General purpose 2d and 3d plot
  - Allows users to select axis values from dropdown. 
  - Clicking a point shows image and data about that run.
  - View runs from 3d? 

## Score Analysis
- New graphs
  - Comparing same comparison method between different galaxy pairs
  - Come up with correlation statistics

- Gather images with same human score and different tiered machine scores
  - Save them in unique folder and write file explaining what comparison, sdss, etc

- SVD on images and see which show most variance>? 

