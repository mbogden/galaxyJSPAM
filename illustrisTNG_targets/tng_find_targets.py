#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
File: tng-find-targets.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden
Created: 2023-Nov-09

Description: 
    This code is designed to interact with the IllustrisTNG Simulation Data. 
    It's goal is to identify close interactions/mergers between two galaxies,
    then saves catalog data relating to the encounter in a csv file.

References:  
- [Add IllustrisTNG ref]
- Sections of this code were enhanced with the assistance of ChatGPT made by OpenAI.

"""


# # Finding Galaxy Mergers within the IllustrisTNG Simulation

# ## Imports

# In[2]:


# ================================ IMPORTS ================================ #
import os, argparse, h5py
import numpy as np, pandas as pd, scipy.signal
import matplotlib.pyplot as plt 
import illustris_python as il

print("Imports Done")

# Global variables
SIM_DIR = '../sims.TNG/TNG50-1/output/'

# A useful fucntion I often use for indented printing
def tabprint( printme, start = '\t - ', end = '\n' ):
    print( start + str(printme), end = end )


# ---
# ## Command Line Arguments
# 
# This is written in JupyterLab, and will be compiled and ran in python for faster execution.  This will define the possible input command line arguements.
# 
# 
# WARNING:  I have not been consistent with implementing and following arguments.  Code still in indevlopment.  

# In[3]:


# This argument decides if code is in python or jupyter.
buildEnv = False

# Define argument parser function 
def initParser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument( '-s', '--simDir', default = '../sims.TNG/TNG50-1/output/',  type=str, \
                        help="Base directory for a single simulation on the IllustrisTNG servers.")   
    
    parser.add_argument( '-n', '--simName', default = 'TNG50-1',  type=str, \
                        help="Name for the simulation being worked on.")
    
    parser.add_argument( '-o', '--overwrite', default = False,  type=bool, \
                        help="Overwrite output files?  If false, will check if output file exists before beginning time-consuming tasks.")
    
    parser.add_argument( '-t', '--trim', default = -1,  type=int, \
                        help="Default number of subhalos to consider, sorted by highest mass first.")
    
    parser.add_argument( '-f', '--function', default = 'None', type=str, \
                        help="Default function program will be executing.")
    
    parser.add_argument( '-d', '--dataDir', default = 'data', type=str, \
                        help="Default location to store misc data files.")

    return parser

parser = initParser()
print("Args: Defined")


# ## To Python? Or to JupyterLab? 
# This will establish if this is being run in a JupyterLab environment or from Command Line in Python. 
# 
# NOTE:  If you're running this in Jupyter, modify the `cmdStr` below to whatever variables you need.

# In[4]:


# Am I in a jupyter notebook?
try:
    
    # This command is NOT available in a python script
    get_ipython().__class__.__name__
    buildEnv = True
    print ("In Building Environment")
    
    # Command Line Arguments
    cmdStr  = 'python3 targets-working.py'
    cmdStr += ' --trim 10'
    cmdStr += ' --dataDir data'
    
    # Read string as if command line
    print( "CMD Line: \n\t$:", cmdStr)
    
    # This function doesn't like the 'python3 file.py' part.
    args = parser.parse_args(cmdStr.split()[2:])

# Or am I in a python script?
except:
    
    # Read CMD arguments
    args = parser.parse_args()
    

print( "Args: Read")
print( args )

# Setup data directory if not found
os.makedirs(args.dataDir, exist_ok=True)


# In[5]:


if buildEnv: 
    # Location of one simulation
    print("Is this locational valid?")
    print( f"Simulation data: {os.path.exists( args.simDir )} - {args.simDir}" )


# ---
# # Halos and SubHalos
# Within the simulation, Halos are the largest set of objects that are gravitationally bound to each other, I like to think of them as galaxy clusters.  Subhalos are also gravitationally bound objects but more dense, and I suspect has to do with potential energy.   I like to think of them as galaxies, globular clusters, blobs of gas, etc.  (That's my reasoning and I'm sticking to it)
# 
# 
# For more information, pleas visit the IllustrisTNG Data Specification Page.  https://www.tng-project.org/data/docs/specifications/

# In[6]:


# Useful function for constructing/deconstructing subhalo ids.

def generate_subhalo_id_raw(snap_num, subfind_id):
    # Convert input to integers in case they are passed as strings
    snap_num = int(snap_num)
    subfind_id = int(subfind_id)
    # Calculate the SubhaloIDRaw
    subhalo_id_raw = snap_num * 10**12 + subfind_id
    return subhalo_id_raw

def deconstruct_subhalo_id_raw(subhalo_id_raw):
    # Convert input to integer in case it is passed as a string
    subhalo_id_raw = int(subhalo_id_raw)
    # Extract SnapNum and SubfindID from SubhaloIDRaw
    snap_num = subhalo_id_raw // 10**12
    subfind_id = subhalo_id_raw % 10**12
    return (snap_num, subfind_id)


# ---
# ## Mass Filter
# 
# So I am looking for larger galaxies that visualize well.  I will be choosing galaxies that are between masses of 1/10th and x10 the Milky Way galaxies.

# In[7]:


n_subhalo = -1

def getMassFilter( args, snapNum, mScale = 10 ):
    
    # This is the first time I pull data for every single subhalo.  Let's save the value for a later time.
    global n_subhalo
    
    # Define where file will be saved
    mLoc = f'{args.dataDir}/{args.simName}-{snapNum}-mask-mass-{mScale}.npy'
    
    # Read from file if it exits
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Mass Mask: {mLoc}")
        mass_mask = np.load( mLoc )
        n_subhalo = mass_mask.shape[0]
        return mass_mask
    
    # define mass limits
    milky_way_mass = 150.0  # in (10^10 M_⊙) 
    upper_mass = milky_way_mass * mScale
    lower_mass = milky_way_mass / mScale
    
    # Pull masses for all subhalos in snapshot
    fields = ['SubhaloMass']
    print("Pulling Masses for all Subhalos")
    print("WARNING: May take a while ")
    SubhaloMass = il.groupcat.loadSubhalos( args.simDir, snapNum, fields=fields)
    
    # This is the first occasion where I wi
    
    # Find galaxies between upper and lower mass
    mask_mass = ( SubhaloMass[:] <= upper_mass ) & ( SubhaloMass[:] >= lower_mass )
    
    # Save mass
    np.save( mLoc, mask_mass )
    
    # Needed elsewhere
    n_subhalo = mask_mass.shape[0]
    
    return mask_mass
    
if buildEnv and True:
    args.overwrite = False
    mask_mass = getMassFilter( args, 67 )
    print( mask_mass.shape, mask_mass.dtype )


# ___
# ## Centrals and Satellites
# Halo's often have a central galaxy that's the largest, with smaller subhalos orbiting it called satellites.  For convenience, let's create a mask of these central galaxies.
# 

# In[8]:


def expand_mask_from_list( true_list ):    
    mask = np.full( n_subhalo, False, dtype=bool )    
    mask[true_list] = True    
    return mask
    

def getCentralFilter( args, snapNum = 99 ):
    
    mLoc = f'{args.dataDir}/{args.simName}-{snapNum}-mask-central.npy'

    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Central Galaxy file: {mLoc}")
        mask_central = np.load( mLoc )
        return mask_central

    print(f"Getting Central SubHalo IDs for sim/snapshot: {args.simName} / {snapNum}")

    # The GroupFirstSub is the subhalo id for the largest subhalo in a halo.  
    GroupFirstSub = il.groupcat.loadHalos( args.simDir, snapNum, fields=['GroupFirstSub'])

    # Filter out groups that contain no subhalos.
    w = np.where(GroupFirstSub >= 0) # value of -1 indicates no subhalo in this group
    central_ids = GroupFirstSub[w]
    
    # Expand into a full array with a value for every subhalo
    mask_central = expand_mask_from_list( central_ids )
    
    # Save mass
    np.save( mLoc, mask_central )
    
    return mask_central

if buildEnv and True: 

    args.overwrite = False
    mask_central = getCentralFilter( args, snapNum = 67 )
    print('Central Galaxies:', mask_central.shape, mask_central[:10] )
    


# # (r) Galaxy Morphologies (Deep Learning)
# 
# Because our method relies on disks of galaxies, it might be useful for us to find mergers betweeen two disk galaxies. 

# In[9]:


# A function to print the upper level of an HDF5 file.
def print_HDF5_info( file_path ):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        print( f"HDF5 file: {file_path}" )
        print("Top-level headers and sizes:")

        # Iterate over items in the root of the file
        for key in file.keys():
            # Get the object (could be a group or dataset)
            item = file[key]

            # Check if the item is a group or dataset and print its size
            if isinstance(item, h5py.Group):
                print(f"\tGroup: {key}, Number of items: {len(item)}")
            elif isinstance(item, h5py.Dataset):
                print(f"\tDataset: {key}, Shape: {item.shape}")
    # Close file

def getDiskMorphologyFilter( args, snapNum = 99 ):
    
    mLoc = f'{args.dataDir}/{args.simName}-{snapNum}-mask-disk-morphology.npy'

    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Disk Morphology Mask: {mLoc}")
        mask_disk = np.load( mLoc )
        return mask_disk
    
    # Check if morphology file exists.  
    hdf5Loc = f'{args.dataDir}/TNG50-1-morphologies_deeplearn.hdf5'
    if not os.path.exists( hdf5Loc ):
        print("File missing: ", hdf5Loc )
        raise AssertionError
    
    # Read the deeplearning morphology file
    with h5py.File(f'subcatalogs/TNG50-1-morphologies_deeplearn.hdf5', 'r') as file:
        
        header = f'Snapshot_{snapNum}'
        
        # Verify snapshot header is in file
        if header not in file.keys():
            print(f"Bad HDF5 header: {header} / {file.keys()}" )
            return None       
        
        subhaloIDs      = np.array( file[header]['SubhaloID'] )
        subhaloDiskProb = np.array( file[header]['P_Disk'] )
        
        # Iterate through and grab subhalos with a greater chance of being a disk galaxy
        disk_list = []
        for i in range( subhaloIDs.shape[0] ):
            if subhaloDiskProb[i] > 0.5:
                disk_list.append( subhaloIDs[i] )
        
    # Done reading file.
    
    # create mask 
    mask_disk = expand_mask_from_list( np.array( disk_list ) )
    
    # Save mass
    print(f"Saving Disk Morphology Mask: {mLoc}")
    np.save( mLoc, mask_disk )
    
    return mask_central    
if buildEnv and True:
    print_HDF5_info( 'subcatalogs/TNG50-1-morphologies_deeplearn.hdf5' )
    args.overwrite=False
    mask_disk = getDiskMorphologyFilter( args, 67 )  
    
    print('Disk Galaxies:', mask_disk.shape, mask_disk[:10] )


# ## (y) Merger History
# 
# Because manually detecting major mergers in the merger tree is messy (trust me, I tried), I'll be using someone else's subcatalogs to detect major mergers between galaxies.   

# In[10]:


def getMajorMergerMask( args, snapNum = 67, snapCutoff=13 ):
    
    mLoc = f'{args.dataDir}/{args.simName}-{snapNum}-mask-major-merger-{snapCutoff}.npy'
    
    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Upcoming Major Merger Mask: {mLoc}")
        mask_merger = np.load( mLoc )
        return mask_merger
    
    file_loc = f'subcatalogs/MergerHistory_0{snapNum}.hdf5'
    
    print( f"Merger History Loc: {file_loc}" )
    print( f"File found: { os.path.exists( file_loc ) }" )
    
    # Return None if no file found for snap num.
    if not os.path.exists( file_loc ):
        print(f"WARNING:  Could not find file: {file_loc}")
        raise ValueError(f"Subcatalog File Missing: {file_loc}")
        return None
    
    # Read Merger History file.    
    with h5py.File(file_loc, 'r') as file:
            
        # Get the object (could be a group or dataset)
        dataset = file['SnapNumNextMajorMerger']
        
        # Create a boolean mask for values that are non-negative and below the upper limit
        mask_merger = (dataset[:] >= 0) & (dataset[:] <= (snapNum + snapCutoff) )
        
    # Saving
    print(f"Saving Major Merger Mask: {mLoc}")
    np.save( mLoc, mask_merger )
       
    return mask_merger
            
    # Find merger
    
# Get merger tree catalog


if buildEnv and True:
    
    # print_HDF5_info( f'subcatalog/MergerHistory_0{snapNum}.hdf5' )
    
    mask_merger = getMajorMergerMask( args, 67, 13 )
    
    print('Disk Galaxies:', mask_merger.shape, mask_merger[:10] )
    
        


# ## (t) Galaxy Morphologies (Kinematic) and Bar Properties
# We would like to know additional details about the distributions of mass bewteen the bulge, discs, and halos.  This subcatalog appears to have that info. 
# 
# NOTE: Ignoring for now.  But potentially a future consideration.

# In[11]:


if buildEnv and False:
    print_HDF5_info( 'subcatalogs/morphs_kinematic_bars.hdf5' )


# ---
# ## Combine Masks Together
# 

# In[12]:


def combine_masks(mask_list):
    
    # Verify that all masks have the same shape
    if not all(mask.shape == mask_list[0].shape for mask in mask_list):
        raise ValueError("ERROR: Combine Masks: All masks must have the same shape")

    # Initialize the combined mask with the first mask
    combined_mask = mask_list[0].copy()

    # Perform logical AND operation with each subsequent mask
    for mask in mask_list[1:]:
        combined_mask &= mask

    return combined_mask

def generate_mask( args, snapNum, mass = True, massScale = 10, central = True, disk = True, major = True, majorCutoff = 13 ):
    
    # Create list of mask to find goi
    mask_list = []
    if mass:     mask_list.append( getMassFilter           ( args, snapNum, mScale = massScale ) )
    if central:  mask_list.append( getCentralFilter        ( args, snapNum ) )
    if disk:     mask_list.append( getDiskMorphologyFilter ( args, snapNum ) )
    if major:    mask_list.append( getMajorMergerMask      ( args, snapNum, majorCutoff ) )

    try:
        # Get 
        combined_mask = combine_masks( mask_list )
        goi_ids = np.where( combined_mask )        
        return goi_ids[0]
        
    except ValueError as e:
        print(e)
        return None

if buildEnv and True:
    
    # Create list of mask to find goi
    mask_list = []
    mask_list.append( getMassFilter( args, 67 ) )
    mask_list.append( getCentralFilter( args, 67 ) )
    mask_list.append( getDiskMorphologyFilter( args, 67 ) )
    mask_list.append( getMajorMergerMask( args, 67 ) )

    try:
        # Get 
        combined_mask = combine_masks( mask_list )
        print(combined_mask.shape)
        print( f"Remaining Subhalos: {np.sum( combined_mask )}" )
        goi_ids = np.where( combined_mask )
        print( f"GOIs: {goi_ids[0].shape}" )
        
    except ValueError as e:
        print(e)

    # Initial GOIs of interest
    goi_ids = generate_mask( args, 67 )
        


# --- 
# ## Find GOI's in Future Merger Trees
# 
# Since requesting a merger tree only returns it's tree for the current moment and backwards in time, I need to jump several snapshots forward, and identify which galaxies it belongs to there.  It's a long tedious process but I'll figure it out.
# 
# WARNING: Something is wrong.

# In[13]:


def generate_subhalo_id_raw(subfind_id, snap_num, ):
    # Convert input to integers in case they are passed as strings
    snap_num = int(snap_num)
    subfind_id = int(subfind_id)
    # Calculate the SubhaloIDRaw
    subhalo_id_raw = snap_num * 10**12 + subfind_id
    return subhalo_id_raw


def reverse_SubhaloIDRaw( SubhaloIDRaw ):
    snapNum = SubhaloIDRaw // 10**12
    subhaloID = SubhaloIDRaw % 10**12
    return snapNum, subhaloID

def find_gois_in_tree( tree_id, snapNum, goi_list, args ):
    
    # Load only SubhaloIDRaw for effecient retrieval time
    tree_subhaloIDRaw = il.sublink.loadTree( args.simDir, snapNum, tree_id, fields=['SubhaloIDRaw'] )

    # See if any of my GOIs are in list
    goi_mask = np.isin( goi_list, tree_subhaloIDRaw )
    n_matches = np.sum(goi_mask)


    # If none found
    if n_matches == 0:
        return []

    # Else we have results
    # print( f'Tree - Matches: {tree_id} - {n_matches}\n' )

    # Get the index locations where the mask is True
    goi_loc = np.where(goi_mask)[0]
    goi_ids_in_tree = goi_list[ goi_loc ]

    goi_tree_list = []

    for goi in goi_ids_in_tree:
        goi_tree_list.append( ( goi, generate_subhalo_id_raw( tree_id, snapNum ) ) )

    return goi_tree_list

def getMOI_v1( args, start_snapNum, stop_snapNum ):
    
    mLoc = f'{args.dataDir}/{args.simName}-moi-list-v1-{start_snapNum}-{stop_snapNum}.txt'
    
    # If already obtained, read from file
    if os.path.exists( mLoc ) and not args.overwrite:
        print(f"Reading Merger-of-Interest List: {mLoc}")
        moi_list = np.loadtxt( mLoc, dtype=int )
        return moi_list
    
    goi_ids = generate_mask( args, start_snapNum )
    goi_ids_raw = np.array([ generate_subhalo_id_raw( goi, start_snapNum ) for goi in goi_ids ])
    print( f"Search for Galaxies of Interest: {goi_ids_raw.shape}")
    
    merger_ids = generate_mask( args, stop_snapNum, mass = True, massScale = 12, central = True, disk = False, major=False )
    print( f"Searching within Merger Trees: {merger_ids.shape}" )
    
    moi_list = []
    for i, mid in enumerate(merger_ids):  
        tabprint( f" {i} / {merger_ids.shape[0]} - {mid}", end='\r' )
        moi_list.extend( find_gois_in_tree( mid, stop_snapNum, goi_ids_raw, args ) )
    
    print( f"\nFound GOI / Tree Matches: {len( moi_list) }")
    
    # Save list for future reference
    moi_list = np.array( moi_list, dtype=int )
    np.savetxt( mLoc, moi_list, fmt='%i', header='merger-goi tree-goi' )
    return moi_list

if buildEnv and True:  
    
    args.overwrite = False
    
    moi_list = getMOI_v1( args, 67, 75 )
    
    for i in range( moi_list.shape[0] ):
        print( i, moi_list[i] )
        


# In[14]:


# Define print fucntion for a row
def printRow( tree, i, fields ):
    # if i == -1:
    #     print("Invalid index")
    #     return
    
    print( " - ".join( [ f"{key}:{tree[key][i]}" for key in fields ]) )
    
def createVisLink( subhaloIDRaw, projection = 'face', simulation='TNG50-1' ):
        tmp = reverse_SubhaloIDRaw( subhaloIDRaw )    
        
        if projection == 'face':
            link = f"https://www.tng-project.org/api/{simulation}/snapshots/{tmp[0]}/subhalos/{tmp[1]}/vis.png?partType=stars&partField=stellarComp-jwst_f200w-jwst_f115w-jwst_f070w&size=1&method=histo&rotation=face-on&plotStyle=edged"
        # Else, project x,y plane
        else:
            link = f"https://www.tng-project.org/api/{simulation}/snapshots/{tmp[0]}/subhalos/{tmp[1]}/vis.png?partType=stars&partField=stellarComp-jwst_f200w-jwst_f115w-jwst_f070w&size=1&method=histo&nPixels=256%2C256&axes=0%2C1&plotStyle=edged"
        return link
    
def analyze_MOI( args, gois ):
    
    mGOI = gois[0]
    tGOI = gois[1]
    
    print( 'Final Viz: ', createVisLink( tGOI ) )
    
    tabprint( f'Merger GOI: {mGOI}' )
    tabprint( f'Tree   GOI: {tGOI}' )    
    
    # If matches found, load more info
    fields = ['SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloGrNr',\
              'SubhaloIDRaw','SubhaloMass', 'RootDescendantID', 'SnapNum', 'DescendantID',\
              'SubhaloPos', 'SubhaloVel', 'SubhaloSpin', 'SubhaloHalfmassRad', ]
    
    tree_snapNum, tree_subhaloID = reverse_SubhaloIDRaw( tGOI )
    goi_snapNum, goi_subhaloID = reverse_SubhaloIDRaw( mGOI )
    
    
    # Load Tree with desired info
    tree = il.sublink.loadTree( args.simDir, tree_snapNum, tree_subhaloID, fields=fields)
    
    # Create a dictionary to map Subhalo IDs to their index in the list
    subhalo_index = {subhalo_id: index for index, subhalo_id in enumerate(tree['SubhaloID'])}  
    ci = 0  # Starting index of requested subhalo/snapshot
    
    # Print some start info for familiarization
    if False: 
        print("Printing basic info for familization")
        print("\nChild Info")
        printRow( tree, ci, fields )

        print("\nPrimary Info")
        pi = subhalo_index.get( tree['FirstProgenitorID'][ci], -1 )
        printRow( tree, pi, fields )

        print("\nSecondary Info")
        si = subhalo_index.get( tree['NextProgenitorID'][pi], -1 )
        if si == -1:
            print("No Secondary Parent")
        else:
            printRow( tree, si, fields )
    
    
    # Grab ids and velocity arrays of the primary galaxies throughout time. 
    pVel = np.ones( (tree_snapNum+1, 3) ) * np.nan    # Velocities
    pIDRaw = np.zeros( ( tree_snapNum+1), dtype=int)  # SubhaloIDRaw
        
    while ci != -1:
        
        i, tmp = reverse_SubhaloIDRaw( tree['SubhaloIDRaw'][ci] )
        
        # Grab array values
        pVel[i,:] = tree['SubhaloVel'][ci][:]
        pIDRaw[i] = tree['SubhaloIDRaw'][ci]
        
        # Update to primary parent in previous snapshot
        ci = subhalo_index.get( tree['FirstProgenitorID'][ci], -1 )

    # Calculate the change in velocity (Δv)
    dVel = np.diff(pVel, axis=0)
    
    # Calculate magnitude of acceleration at each time step (assumption Δt=1)
    pAcc = np.sqrt( np.sum( dVel**2, axis=-1 ) )
    snapshots = [ reverse_SubhaloIDRaw( pid )[0] for pid in pIDRaw ]
    
    plt.xlim( 50, 75 )
    plt.plot( snapshots[1:], pAcc )
    
    # Grab peaks in acceleration after snapshot
    cSnapshot = 55 
    
    # Find peaks and their prominences
    peaks, properties = scipy.signal.find_peaks(pAcc[cSnapshot:], prominence=True)
    peak_snapshots = peaks + cSnapshot + 1
    prominences = properties['prominences']
    
    # Print or use the sorted peaks and their prominences
    idList = []
    print("")
    for i, sn in enumerate(peak_snapshots):
        if prominences[i] < 5: continue
        plt.axvline(x=peak_snapshots[i], color='r', linestyle='--', label=f'{peak_snapshots[i]} - {prominences[i]:.2f}')
        print(f"Peak at index {peak_snapshots[i]} with prominence {prominences[i]}: {pIDRaw[peak_snapshots[i]]} - {pIDRaw[peak_snapshots[i]+1]}")
        
        
        for j in range( peak_snapshots[i]-1, tree_snapNum, 1):
            link = createVisLink( pIDRaw[j] ) 
            tabprint( link )
        
    data = {}
    
    plt.axvline( x=75, color='k', linestyle='--' )
    
    plt.xlabel('Snap Shots')
    plt.ylabel('Acceleration Magnitude')
    plt.title(f"Fly-by Detection for SubhaloIDRaw: {tGOI}")
    plt.legend()
    
    return tree
    

if buildEnv and False:  
    moi_list = getMOI_v1( args, 67, 75 )
    
    tmp_tree = analyze_MOI( args, moi_list[51] )
    
    print( tmp_tree.keys() )


# In[15]:


def get_moi_info( args, gois ):
    
    mGOI = gois[0]
    tGOI = gois[1]
        
    # tabprint( f'Merger GOI: {mGOI}' )
    # tabprint( f'Tree   GOI: {tGOI}' )
    
    # If matches found, load more info
    fields = ['SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloGrNr',\
              'SubhaloIDRaw','SubhaloMass', 'RootDescendantID', 'SnapNum', 'DescendantID',\
              'SubhaloPos', 'SubhaloVel', 'SubhaloSpin', 'SubhaloHalfmassRad', ]
    
    tree_snapNum, tree_subhaloID = reverse_SubhaloIDRaw( tGOI )
    goi_snapNum, goi_subhaloID = reverse_SubhaloIDRaw( mGOI )    
    
    # Load Tree with desired fields
    tree = il.sublink.loadTree( args.simDir, tree_snapNum, tree_subhaloID, fields=fields)
    
    # Create a dictionary to map Subhalo IDs to their index in the list
    subhalo_index = {subhalo_id: index for index, subhalo_id in enumerate(tree['SubhaloID'])}  
    ci = 0  # Starting index of requested subhalo/snapshot
    
    # Grab ids and velocity arrays of the primary galaxies throughout time. 
    pVel = np.ones( (tree_snapNum+1, 3) ) * np.nan    # Velocities
    pIDRaw = np.zeros( ( tree_snapNum+1), dtype=int)  # SubhaloIDRaw
        
    while ci != -1:
        
        i, tmp = reverse_SubhaloIDRaw( tree['SubhaloIDRaw'][ci] )
        
        # Grab array values
        pVel[i,:] = tree['SubhaloVel'][ci][:]
        pIDRaw[i] = tree['SubhaloIDRaw'][ci]
        
        # Update to primary parent in previous snapshot
        ci = subhalo_index.get( tree['FirstProgenitorID'][ci], -1 )

    # Calculate the change in velocity (Δv)
    dVel = np.diff(pVel, axis=0)
    
    # Calculate magnitude of acceleration at each time step (assumption Δt=1)
    pAcc = np.sqrt( np.sum( dVel**2, axis=-1 ) )
    snapshots = [ reverse_SubhaloIDRaw( pid )[0] for pid in pIDRaw ]
    
    snaploc = 1000000000000
    
    data = {}
    
    for snapFind in range( 55, 75 ):
        #print("#####   %d   #####" % snapFind)
        
        snapnum_mask = (tree['SubhaloIDRaw'] // snaploc) % snaploc == snapFind
        snapnum_index = np.where( snapnum_mask )

        # Gather masses
        snapnum_masses = tree['SubhaloMass'][snapnum_index]

        # Find n highest masses
        n = 2
        top5_index = np.argsort(snapnum_masses)[-n:][::-1]
        
        if len(top5_index) <= 1: continue
       
        pid = snapnum_index[0][top5_index[0]]
        sid = snapnum_index[0][top5_index[1]]
        
        keys =  [ 'SubhaloIDRaw', 'SubhaloMass', 'SubhaloPos', 'SubhaloVel', 'SubhaloSpin', 'SubhaloHalfmassRad', ]
        
        data[snapFind] = {}
        data[snapFind]['p_acceleration'] = pAcc[snapFind]
        
        for k in keys:
            for c, ii in [ ('p',pid), ('s',sid) ]:
                #print( k, c, ii )
                data[snapFind]['%s_%s'%(c,k)] = tree[k][ii]
                
        data[snapFind]['xy_projection'] = createVisLink( tree['SubhaloIDRaw'][pid], projection = 'xy' )
        data[snapFind]['p_face_projection'] = createVisLink( tree['SubhaloIDRaw'][pid], projection = 'face' )
        data[snapFind]['s_face_projection'] = createVisLink( tree['SubhaloIDRaw'][sid], projection = 'face' )
    
    return data
        

def save_moi_info( args, moi_list, start_snapnum, stop_snapnum ):
    
    fLoc = f'{args.dataDir}/{args.simName}-moi-info-{start_snapnum}-{stop_snapnum}.csv'
    
    # If file exists, read and return.
    if os.path.exists( fLoc ) and args.overwrite == False:
        df = pd.read_csv( fLoc )
        return df
    
    # Else, create file by getting info via merger trees.
    data = {}
    n = len( moi_list )
    for i in range( n ):
        print( i, ' / ', n, end='\r'  )
        data[moi_list[i][0]] = get_moi_info( args, moi_list[i] )
    print('')
    # Convert the nested dictionary to a list of records
    records = [{'moi_SubhaloIDRaw': subhalo_id, 'snapnum': snapnum, **props}
               for subhalo_id, snaps in data.items()
               for snapnum, props in snaps.items()]

    df = pd.json_normalize(records, sep='_')
    print( df )

    df.to_csv( fLoc , index=False )
    
    return df
    

import pandas as pd
if buildEnv and True:  
    
    start_sn = 67
    stop_sn = 75
    
    moi_list = getMOI_v1( args, start_sn, stop_sn )
    df = save_moi_info( args, moi_list, start_sn, stop_sn )
    
    df_filtered = df
    
    # Let's do some filtering for good targets.    
    print( df.columns )
    moi_list = df['moi_SubhaloIDRaw'].unique()
    
    for moi in moi_list:
        #print( moi )
        
        m_condition = df['moi_SubhaloIDRaw'] == moi
        p_mass = df.loc[m_condition, 'p_SubhaloMass']
        s_mass = df.loc[m_condition, 's_SubhaloMass']
        mass_ratio = s_mass / ( p_mass + s_mass )
        ratio_cutoff = 0.1
        ratio_condition = mass_ratio > ratio_cutoff
        
        if not np.any( ratio_condition ):
            df_filtered = df_filtered[ df_filtered['moi_SubhaloIDRaw'] != moi ]
        
    print( df_filtered['moi_SubhaloIDRaw'].unique() )
        
    # First, let's grab all the GOIs to loop through.

    df_filtered.to_csv('TNG50-1-moi-final-v1-67-75-10.csv')


# In[16]:


# Read HDF5 file

import h5py

def print_structure_and_size(file_name):
    with h5py.File(file_name, 'r') as file:
        def print_info(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {node.shape}, Data type: {node.dtype}")
            else:
                print(f"Group: {name}")

        file.visititems(print_info)

file_loc = '../sims.TNG/TNG50-1/postprocessing/tracer_tracks/tr_all_groups_99_meta.hdf5'
print_structure_and_size(file_loc)
print('')
print_structure_and_size( '../sims.TNG/TNG50-1/postprocessing/SubboxSubhaloList/subbox1_67.hdf5' )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # DEPRECATED
# 
# Everything below this line is old and for reference purposes only...

# # Current Seletion Plan.  
# Since there are millions of galaxies, a hundred snapshops, and many millions of possible mergers, I'm going to try and do a selection screening for ideal merger events.  Here are the basic steps I'm considering
# 
# 0) (DONE) Select Snapshot with available subcatalog info.
#     - Status: Using Snapshot 67
#     - Thus far, the Morphology Subcatalog only has data for a few snapshots, therefore we will focus on these snapshots.
#     - If this proves to be too limited, we can build our own deep learning model, training on the X:Kinematic and Y:Galaxy DL Morphologies.
#     
# 0) (Done) Use "Merger History" subcatalog to find galaxies about to undergo a major merger.
#     - Status: Found subhaloID's for 67 about to undergo a major merger within 7 snapshots.  
#         - Hencefort called MOI - (Mergers of Interest) 
#     - Use "SnapNumNextMajorMerger" and filter to X snapshots in future.
#     - Return SubhaloID's
#     - Collect all snapshots into a single list
#     
# 0) (Working) Use Merger Tree Catalog to get additional info
#     - Status:  
#         - Looking into subhalo 0 in snapshot 75 for moi's.  Found 68
#         - Working on pulling data of interest for those moi's. 
#     - NOTE: Requesting a Merger Tree with a SubhaloID and Snapshot, will return it's merger tree UP TO THAT SNAPSHOT and sadly not into the future.
#         - Therefore we will randomly (sorted by biggest mass first) look at merger trees in the final snapshot.
#         - We will then search those mergers trees to see if they contain any of our galaxies of interest.  
#         - Since they're giant lists, this shouldn't take tooooooo long...  hopefully...  (fingers crossed)
#     - Optional 
#         - Filter based on Mass threshold
#             - 1/4 the Milky Way and above
#         - Filter based on Central Galaxy.
#     - When galaxy of interest is found.
#         - Grab Halo ID.
#         - Go forward in time to retrieve ID of secondary galaxy
#         - Go backwards in time to collect velocity info
#         
# 0) Use "Morphological Deep Learning" Subcatalog to get morphology type of both parents.
#     - Using ID of primary and secondary
#     
# 0) Use Subhalo Catalog and Kinematic subcatalog to retrieve orbital parameters of both galaxies
#     
# 6) Using Visualization tool, search for Halo ID and Snapnum. 
# 
# 0) Feed the orbital parameters we find in IllustrisTNG, into SPAM.  Does it create an image (tidal features) similar to Illustrng visual? 
# 
