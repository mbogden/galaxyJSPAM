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
    - [w] Git access on TNG server.
    - [w] Find potential targets: Mergers-of-interest (moi)
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

    - [x] Get images
        - [x] Preliminary automated URL
            - [w] Duos: 50 - 67
            - [ ] Duos: 67 - 99
            - [ ] All: 50 - 67
            - [ ] All: 67 - 99
        - [ ] Modify viewing angle and update collision parameters.
        - [ ] Standard visualization.  Ex SDSS, JWST, wavelenths, etc. 


- Comparing SPAM to TNG
    - [x] Unit conversion between simulations
        - [x] Verify unit conversion is working correctly.   
    - [x] Preliminary SPAM runs on TNG target parameters
        - NOTE: Images do NOT look similar. (lead to historical affiliation above)
    - [x] Unit Conversation error?  (Working properly)
    - [x] Dynamical Friction? (Found and modified to work properly)
        - NOTE: Use lnl = 0.15 for best best of tng-target: 67000000350284    
    - [x] View TNG pts over time
    - [x] Viewing particles Together 
    - [x] Look at orbits!  Do orbits of TNG and SPAM match?
        - NOTE: They do not match with default SPAM settings
        - [x] Do simple velocity projection of 2nd galaxy  
    - [w] Do particles have similar Tidal Features?
      - [ ] Create plot for both particles?
      - [ ] Do initial disks relatively align?
      - [ ] Do tidal features match? 
      
      
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

- Look into WandB for DL metrics




4000 particles
256 generations
256 pop size
