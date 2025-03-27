# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
*Might* be good for others to view.
    
# Matt's Working To-Do
[ ]: Not Started
[w]: Work in progress
[d]: Draft - Partially working
[x]: Complete - Fully working

- Big Picture:
  - WOrking on 'spam_tuner_tng_targets.py' in SPAM.
  - Run Optimizer on TNG targets to fine tune spam params.

- Simulator
  - [d] Custom_runs: Working draft 
    - NOTE: both init and final particles are off.
    - [x] Figure out why pts are off by a single timestep for init and final particles.  Look at code between initializing particles and writing them.
      - NOTE: Found hard-coded tStart = -5 causing issue.  Modified to 0.
  - [d] Rewrite main_simulator.py to use custom_runs with new SIMR pipeline
  - [ ] Change simulation stopping condition to go back further in time.
    - NOTE: I suspect it's designed to find the moment-of-closest-approach (MOCA) and double the time between start and MOCA.
            Going back further would improve SPAM hyper parameter search and scoring function.  
  - [ ] Add many endings.
  - [ ] Encorporate into SIMR pipeline
  - [ ] Look up Allen Harvey Dissertation: A Pipeline for Constructing Optimized N-Body Models of Interacting Galaxies
  - [ ] Modify Dynamical Friction in SPAM
      - Wiki Explanation: https://en.wikipedia.org/wiki/Dynamical_friction
      - Maxwell's Distribution
      - lnl is typically between 1 and 10.
      - look up Coulomb logarithm.
    - [x] Find where variable is defined in code
      - FORTRAN LOC: lnl -> df_module.init_distribution -> init_module.create_collision
      - [x] Created input variable to modify dynamical friction
    - [x] Run tests with dynamical friction
        - NOTE: Changing this value does not change simulation....
      - [x] Find where dynamical friction is used in the code
        - FR: lnl -> integrator.diffeq_nbi -> inegrator.wrap_rk4
        - NOTE: DF only considered if you use potential type 'diffeq_nbi'.  There are others.
          0: diffeq_spm
          1: diffeq_nbi
          2: diffeq_mond
      - [x] Find out where and how potential types are chosen
        - [x] Flow of functions (FOF) call to choose potential type
          - FR: integrator.wrap_rk4.potential_type -> init_module.take_a_step -> custom_runs.basic_run
          - NOTE: wrap_rk4 is definitely called.  There was another wrap_rk41.
        - [x] Where is potential type defined?: 
          - SIMR_CUSTOM_COLLISION...
          [x] Was Hard coded to 0.  Changed to 1.
    - [x] Verify dynamical friction is working correctly.

- Illustris TNG (Target Search)
  - [x] Git access on TNG server.
  - [x] Find potential targets: Mergers-of-interest (moi)
      - [x] Preliminary Filters (Moi_1)
          - NOTE: MOI_1 are ideal galaxies predicted to undergo a merger event soon.
          - [x] TNG-50 (most detailed simulation)
          - [x] Mass
          - [x] Morphology (Disk, Elliptical, etc)
          - [x] Merger History
          - [x] Central vs Satellite (Not used at this time)
          - [x] Performed search on snapshots
              - [x] 50 - 67
              - [x] 67 - 99
              - [ ] 60 - 75
                
        - [x] Search For children of merger in Merger Tree (Moi_2)
            - NOTE: MOI_2 are future children who have already undergone a merger. 
            - [x] Load future merger trees and search for MOI_1 parents.
                - [x] Save trees with MOI_1 parents as MOI_2. 
            - [x] Load MOI_2 trees with catalog fields of interest.
            - [x] Predict primary and secondary parent based on mass for each snap.
            - [x] Generate URLs for preliminary images of targets for each snap.
            - [x] Save parent catalog info + url for MOI_2 at each snap.
            - [x] Find Duos: MOI_2 with 2 or more MOI_1 as parents
                - NOTE: 2 or more MOI_1 could indicate both parents of mergers passed filters. 
            - [x] Performed search on snapshots
                - [x] 50 - 67
                - [x] 67 - 99
                - [ ] 60 - 75
                    - NOTE: Found several potential targets at 65-69 range, but didn't have enough surrounding snapshot info to fully analyze.
                    
        - [w] Search through preliminary images (MOI_3)
            - NOTE: Images with potential tidal features are notated as MOI_3.
            - [w] Manually open and identify tidal features, record Snap and SubhaloID.
                - [x] Duos: 50 - 67
                - [ ] Duos: 67 - 99
                - [ ] All: 50 - 67
                - [ ] All: 67 - 99
 
    - [x] Get collision parameters on Potential MOI_3
        - [d] Download particles
        - [x] Create functions calculating needed parameters from particles
            - [x] Verify calculated parameter match catalog parameters.
            - [ ] Cannot figure out while spin magnitudes don't match.  Directions match.
        - [x] Plot particles together
            - NOTE: Found that primary galaxy seems to have "stolen" secondary particles after the flyby event, despite pts being closer/orbiting to secondary still.
        - [x] Use historical affiliations to reassign particle ownership
        - [x] Do parameter calculations on new set of pts based on historical affiliation.
        - 
    - [x] Create Standardized Reference Frames (RFs).
      - [x] Orbital Reference Frame
        - [x] Set the orbital "plane" to be the xy-plane
        - [x] Place galaxies centers on the x-axis.
        - [x] Set the origin as the half way distance between both galaxies.
        - [x] Place angular momentum in the positive z-direction.
      - [x] Tidal features
        - [x] Using PCA, to find plane most likely to show tidal features.
        - [x] Apply to both galaxies, and seperately

    - [x] Create images
        - [x] Preliminary automated URL
            - [w] Duos: 50 - 67
        - [x] Orbital RF image
        - [x] Tidal RF image
          - [x] Both
          - [x] Primary
          - [x] Secondary
        - [w] Spread of angles
          - [w] Automate changing/tilting angles to get images of galaxies from every point of view
      - [x] Use histogram of star particles
        - NOTE: While not the most accurate representation of the galaxy, it does show the general shape of the galaxy.   Thus it does captures tidal features. 
      - [ ] Standard wavelgenth visualization.  Ex SDSS, JWST, wavelenths, etc. 
        - NOTE:  Only certain snapshots have photometric data.   Get list of potentials targets from these snapshots first.  
  - [x] Find targets to anlayze
  - [x] Generated 1200 composite images showing orbital and tidal features.
  - [x] Create UI to "Rate" tidal features for images
    - [x] Finish rating images
  - [x] Review ratings and create target list


- Preliminary Comparing SPAM to TNG
    - [x] Unit conversion between simulations
    - [x] Standardize SPAM and TNG parameter array.
      - [ ] Analyze SPIN and halfmass radius before and after flyby.
    - [x] Get SPAM running (See SPAM notes)
    - [x] Preliminary SPAM runs on TNG target parameters
        - NOTE: Images do NOT look similar
    - [x] Dynamical Friction? (Found and modified to work properly)
        - NOTE: Use lnl = 0.15 for best best of tng-target: 67000000350284    
    - [x] View TNG pts over time
    - [x] Viewing particles Together 
    - [x] Look at orbits!  Do orbits of TNG and SPAM match?
        - NOTE: They do not match with default SPAM settings
        - [x] Do simple velocity projection of 2nd galaxy  
    - [x] Play with following variable to get matching tidal features.
      - SETBACK:  Too many variables to adjust.  Will need to find an optimize method.  
        - NOTE: Based on following: Lars Hernquist. N-body realizations of compound galaxies. The Astrophysical Journal Supplement Series, 86:389{400, June 1993.
        - lnl 
            - NOTE: Value around 0.15 works well.
        - mhalo, rhalo, rchalo (in df_module)
            - NOTE: These values were hardcoded based on Milky Way and M31.
            - NOTE: Chancing rchalo to be bigger leads to better tidal features.
        - velocity
            - NOTE: Slightly slowly down vel (0.9) creates better matches.
        - mass
    - [x] Find targets with matching Tidal features.
        - 

- Convert TNG to SPAM 
  - [x] Converting TNG kinematics to SPAM units
    - [x] v1: Working kinematics to SPAM (Uses only current values)
    - [x] v2: Using past kinematics that might have changed (orientation, radius, mass etc)
  - [x] Implement SPAM testing on TNG parameters
    - [x] Gridsearch through lnl, r_scale, and halo_mass_ratio
    - [x] Create orbits for future anlaysis. 
  - [x] Scoring function centered on moment of closest approach.  
    - [x] Uses guassian curve centered on moment of closest approach.
    - [x] Auto gen guassian_variance based on edges = 0.01.
    - [x] Shorten the best orbit and verify the score changes. 

    - Analyze TNG targets to see if they're similar or drastically different.
      - 3 - 5
    - If many targets have very different ratios, then we may need to add these as variables to optimize upon
    - If they're semi consistent, then perhaps we can optimize them now using TNG targets.  

- Parallel Processing Manager
  - [x] Create MPI general purpose Queue Master/Worker system
    - NOTE: Uses MPI so it can be scaled as needed on cluster for cmdline use.
    - [x] Working draft w/ tests
  - [ ] Create a Multithreading queue manager so jupyter notebook can use it.
  
PAPER 1: Can SPAM recreate realistic tidal features with Artificial Targets?
- [x] Yes, BUT.  I have to change hard-coded variables to re-create tidal features.  These may become  The variables I can change.
  - lnl: A variable that controls the strength of dynamical friction as the galaxies interact.
  - r_scale: Believed to control the size ratio between baryonic matter and the dark matter halo
  - m_halo: Believed to contorl the mass ratio between baryonic matter and the dark matter halo. 
  - NOTE: If the same set of values can work for multiple targets, then it may be as simple as updating the values to be more realistics.  r_scale, and m_halo are based on Milky War measurements from the 90's.  
- [x] Metric for tidal features:   I found that the more the spam orbit and the artificial orbit converges, the more similar the tidal features during interaction.  This makes sense
  - [x] Scoring Metric v1: Avg-error squared between SPAM and TNG orbits.
    - [x] Time-based Avg-error squared between TNG spline and SPAM orbit
    - [x] Nearest-position based avg-error squared between TNG spline and SPAM orbit.  
    - [x] Weight function:  Since the moment-of-closest-approach (moca) is theorized to be the most infulential moment for the tidal features, i created a gaussian weight function centered on the moca.
    - [x] Inital Gridsearch:  Found 2 issues.
      - Degeneracy:  For test target 1, the 3 variables were degenerate for orbit metric v1.  There exists a continuous range of value sets, that have low error.  
      - Pulling orbit to pre-moca.  I noticed the SPAM orbit often diverged  
  - [x] Scoring Metric v2: Pre-MOCA scale
    - NOTE: Based on gridseach, 
- [x] Do grid-search, analyze results
  - [x] Initial grid search

- 


- Optimization Manager
  - [x] Space Manager
    - [x] Define spacial dimension.
    - [x] Define limits
    - [x] Define scale (linera, logithmic)
    - [x] Normalize range (linear & Log)
    - [ ] Set ints or specific values
      - NOTE: Setting specific values such as [ -1, 1 ] can help with symmetries.
  - [x] Execution Manger
    - [x] Integrate with Space Manager
    - [x] Run Black box with Queue Manager
    - [x] Auto change function logger level for convenient prints
    - [x] Reciprical standardizes error function between 0 and 1 nicely. 

  - [ ] Genetic Algorithm
    - [x] Base pyGAD Inputs
    - [x] Get Basic Example Working
    - [d] Get SPAM to TNG Target Tuning working.
      - [x] Works for test target 1
      - [w] Fails terribly for test target 2.  Bad target or bad coding referencing target 1? 
    - [ ] Custom Gene Mask: For variables that are degenerate, it may be useful to keep them together.
      - NOTE: Clumping degenerate variables together, (variables related to symmetry) could improve convergence.
    - [ ] Simulated Annealing Mutation Rate: Because I like the idea of converging to a solution over time.
    - [x] Store dest solutions for later evalution
    - [ ] Spacial Analysis:  Good zones, dependencies, etc. 

- Docker
  - [x] Update beta-3
    - [x] Made notes on how to make an image
    - [x] Updating image to include Astropy
    - [s] Push updated image to DockerHub
    - [x] Make notes on how to update
  - [ ] Update Beta-4: (Builds on beta-3)
    - [ ] Add 'll' alias to bash file.
    - [ ] pip install --upgrade pip
    - [ ] pip install --upgrade nbconvert
    - [ ] pip install scikit-optimize

  - [ ] Slim down Docker Image
    - NOTE:  I grabbed a working Docker image from Dr. Phillips.  Most importantly, it does work.  But it has GB's of unused packages, making it 20+ GB to build... 

- Reorganization
  - [x] utilites
  - [d] model manager
  - [ ] target manager
  - [d] Simulator

- Look into WandB for DL metrics

4000 particles
256 generations
256 pop size


- Commands to remember
    - `python3 main_SIMR.py -targetDir path/to/target -newInfo -newRunInfo -newBase -newRunBase` 
    - `python3 main_SIMR.py -dataDir path/to/data -newScore -newImage -paramName zoo_0_direct_scores`
    -  ``
- For cluster use
  - Create cmdlines of all possible runs, save as scripts in galStuff/runs/whatever_folder/
  