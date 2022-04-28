# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
Might* be good for others to view.

    
- Commands to remember
    - `python3 main_SIMR.py -targetDir path/to/target -newInfo -newRunInfo -newBase -newRunBase` 
    - `python3 main_SIMR.py -dataDir path/to/data -newScore -newImage -paramName zoo_0_direct_scores`

# Matt's Working To-Do

- Create White Paper
    - What can it do now? 
        - modules
        - parameters
        
    - what next steps do I want to take?
        - model optimization
        - comparing and contrasting methods
        - verify results. 

- Capsule Networks

- Reimplemnet WNDCHRM in Galaxy Code.
    - Install WNDCHRM on needed machine. 
    - normalize entire systems.
    - implement mask (Only extract these x out of 1000's of features)
    
- Implement feature selection on WNDCHRM.
    - You did this in Data Mining.  Do it now. 

- Creating Images appears to have a memory leak.  
    - Find and prevent?
        
- Get a single-layer neural network going
    - Gather and pickle images into a temp file/folder.
    - Use GPU to make the creation faster.
    - Generalize it to all targets.
    - Create custom equation and weights in neural network. 

- Create Robust Analysis between Different Score parameters. 
    - Interactive window and grid? 

- Modernize Simuation Code

- Old Stuff
    - Capstone execution times, why are they acting funny?         
        - Redo speed test with only creating new scores. 
    - Test IO choke-point with reading particles.
        - Might not be an issue if you only create images once. 
        - Test how big pickle files are for just X,Y particles.
        - If issue, consider copying files to scratch.
    - Veryify new scores are being created.
    
## Big picture
- Create automatic pipeline for evolution
    - Get function to create status update!
    - Create reports for any missing items. 
    
- Optimization opportunities
    - Parallel processing
        - If one model errors out, the rest of the models the processor is responsible for don't get accomplished.
        - When attempting to repeat the parallel environment, that came processor is in charge of everything that remains to be done, and remaining processors do nothing. 
        - Randomize list being broadcast? 
        - Implement Tasker/Working solution? 
    - Simulation 
        - Analyze impact approximation w/o points. Stop if small 
        - Particle Files: Pickle X,Y,R and zip.  
        - Compare model scores between 100k and 20k pts 
    - Image Maker
        - During mpirun, when 1 run fails, processor does not complete rest.
        - Solution: Implement Tasker/Worker model for parallel processing? 
    - WNDCHRM 
        - implement masks

- Image Creator
    - Total variation denoising on Model images?

- Image Comparison.
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
      - Simulated annealing for pixel difference, ( or any machine method) 
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
      - Take score of all comparison methods and send through machine learning
  - Can I train a NN to take images and predict Orbital parameters? 
      - 
      

- Triple image comparison
	- Compare all Model, Unperturbated, and target image
	- Where Model and unperturbed closely match, lower weight on model image
    - Train DNN on all three images. 

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

# Matt's Personal Setup
- WSL
- Open JupyterLab in GalStuff
- 
# Meeting Notes

## Feb. 2, 2022
### Accomplished
- Color-coded priority by publication
- Read and Commented Graham's dissertation
- Old NN code. 

### Cover
- What Publication stuff should I work on? 

### Questions for Graham in GA code
- mmm, xLims, shrink
    - mmm's appears to be the the SPAM min/maxs of the zoo: merger models converted to GA parameters. 
    - xLim's become based on the "target" model you're choosing.
        - Later, we won't have knowledge on the "target" values

        
    - When a new model population is created, we don't know the psi for those.
        - Your code uses the psiReal during GA run.  We won't know this moving forward right?
        - Does using psiReal limit the range of your model space by a 1/4th? 
        
- PSI.  In real case, when we don't know.  
    - Run 4 GA's with each option.

- MMM's
    - hardcoded.
    - based on top Zoo Merger models. 
        - zooThresh.

- initSeed
    

### Active Working
GA_working
- Find purpose of following variables, rename as needed, save in input file
    - nPop: "Size of population at each step"
    - nParam:  "14" Number of SPAM parameters the GA is creating/using.
    - pFit: "[ 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 ]"  The parameters which GA is actually making changes to. 
    - xLim:  Limits to parameters being fitted.  Currently based on Galaxy Zoo: Merger Files. 
    - mmm: mins and maxes of the Zoo Merger models.
        - Modified into GA parameters? 
        - How do we determine the limits later?  
    - psi:  2-element matrix of values either -1 or 1.
        - Will need to run 4 different simulations to cover all parameter space. 
    getInitPop
        - initType:  two different initializing population types
            - 0: Uniformly random distribution from min to max. 
                - popSol: (nPop x nParam) matrix filled with random value between limits and pReal values. 
            - 1: (hardcoded) linearly spaced values from min to max.
                - Loop 1000 times to find maxCorr and save corresponding R.
                    - R: (nParam X nPop) matrix:  Appears to be distribution of points across parameter space. 
                    - corr: ( det( corr( R ) ) )
                    - Save R with maxCorr found whiling looping 1000 times.
                - Generate popSol: (nPop x nParam)
                    - pFit values get maxR value found
                    - non pFit get pReal values.  (Target values)
                    
    - mixProb
    - stds: Looks like width of SPAM parameter limits.
    - sigScale: 0.05 ("scale param stdev for prop width") Same for all SPAM parameters?!
    - pWidth: stds*sigScale
    - pReal: "data[targetInd, 0:-1]" Am I feeding it the ideal spam parameters?
    - pBest: The best of the previous generation...? 
    - xLim: 
- Changes to make later
    - getInitPop "initType": Consider making this a variable to feed in later.
    
    
# 