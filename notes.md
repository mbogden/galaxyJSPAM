# Notes for galaxyJSPAM
These are written by Matthew Ogden for Matthew Ogden while creating and organizing code.  
*Might* be good for others to view.
    
- Commands to remember
    - `python3 main_SIMR.py -targetDir path/to/target -newInfo -newRunInfo -newBase -newRunBase` 
    - `python3 main_SIMR.py -dataDir path/to/data -newScore -newImage -paramName zoo_0_direct_scores`
    -  ``

# Matt's Working To-Do
[ ]: Not Started
[w]: Work in progress
[d]: Draft - Partially working
[x]: Complete - Fully working

- Simulator
  - [x] Custom_runs: Working draft 
    - NOTE: both init and final particles are off.
    - [x] Figure out why pts are off by a single timestep for init and final particles.  Look at code between initializing particles and writing them.
      - NOTE: Found hard-coded tStart = -5 causing issue.  Modified to 0.
  - [d] Rewrite main_simulator.py to use custom_runs with new SIMR pipeline
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

- Illustris TNG
  - [w] Find potential targets
    - [x] Preliminary Filters
      - [x] TNG-50 (most detailed simulation)
      - [x] Mass
      - [x] Morphology (Disk, Elliptical, etc)
      - [x] Merger History
      - [w] Cosmological Jellyfish
  - [x] Get collision parameters
  - [x] Get images
    - [ ] Automate image generation
    - [ ] Modify viewing angle in a convenient way.
    - [ ] Create images if standard visualization.  Ex SDSS, JWST, wavelenths, etc. 
  - [x] draft tng -> spam unit conversion
  - [x] init SPAM runs of TNG targets
    - NOTE: images do not look correct. Need thorough testing to identify why
    - [x] Look at orbits!  Do orbits of TNG and SPAM match?
      - NOTE: They do not match.
      - [x] Figure out why they don't match
        - [x] Unit Conversation error?  (Working properly)
        - [x] Dynamical Friction? (Found and modified to work properly)
        - NOTE: Use lnl = 0.15 for best best of tng-target: 67000000350284
    - [w] Do particles have similar Tidal Features?
      - [ ] Create plot for both particles?
      - [ ] Do initial disks relatively align?
      - [ ] Do tidal features match? 

- Docker
  - [x] Update beta-3
    - [x] Made notes on how to make an image
    - [x] Updating image to include Astropy
    - [s] Push updated image to DockerHub
    - [x] Make notes on how to update
  - [ ] Update Beta-4: (Builds on beta-3)
    - [ ] Add 'll' alias to bash file.
  - [ ] Slim Docker Image
    - NOTE:  I grabbed a working Docker image from Dr. Phillips.  While it does work, it has GB's of unused packages, making it 20+ GB to build... 

- Reorganization
  - [x] utilites
  - [d] model manager
  - [ ] target manager
  - [d] Simulator

- Comparing SPAM to TNG
  - [x] Unit conversion between simulations
    - [x] Verify unit conversion is working correctly.  
    - [x] Do simple velocity projection of 2nd galaxy
  - [x] Viewing particles Together
  - [x] View TNG pts over time
    - [x] Pull star pts at set snapshots
  - Orbits are OFF!
    - play with mass profiles of galaxies
      - df_module -> ( mhalo, rhalo, rchalo )
      - Make rhalo smaller
      - Look up this: 
        Lars Hernquist. N-body realizations of compound galaxies. The Astrophysical Journal Supplement Series, 86:389{400, June 1993.
  
  - Ideas
    - Have function set the bulge/disk/halo scales/mass ratios.  
      - These values are hardcoded based on Milky Way and M31.
      - Analyze TNG targets to see if they're similar or drastically different.
        - 3 - 5
      - If many targets have very different ratios, then we may need to add these as variables to optimize upon
      - If they're semi consistent, then perhaps we can optimize them now using TNG targets.  

- Look into WandB for DL metrics




4000 particles
256 generations
256 pop size
