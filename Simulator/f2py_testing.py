#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: custom_runs_testing.py
Author: Matthew Ogden
Email: ogdenm12@gmail.com
Github: mbogden 
Created: 2024-Jun-04

Description: This script is for testing and development of the custom_runs fortran module
"""

import os
from os import system, remove, listdir, getcwd, chdir, path, rename
from sys import path as sysPath

# ===================== LOAD PRE EXISTING POINTS ===================== #

if True:
    # For loading in Matt's general purpose python libraries
    supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
    sysPath.append( supportPath )
    import general_module as gm
    import info_module as im

    im.test()

    tDir = '../targetDir/'
    tDir = gm.validPath(tDir)

    tInfo = im.target_info_class( targetDir = tDir, printAll=False)
    if tInfo.status == False:
        print("WARNING: target info class bad")
    else:
        print("Target Good!")

    # Assert target status
    assert tInfo.status == True, "WARNING: Target info class bad"

    # Get run info class
    rInfo = tInfo.getRunInfo( )

    if rInfo.status == False:
        print("WARNING")
    else:
        print("Run '%s' Good!"%rInfo.get('run_id'))

    assert rInfo.status == True, "WARNING: Run info class bad"

    # Get the particles in the run
    ipts_4k, fpts_4k = rInfo.readParticles( '4k' )
    print( ipts_4k.shape, fpts_4k.shape )


# ===================== TEST CUSTOM FORTRAN FUNCTIONS ===================== #

import numpy as np
import matplotlib.pyplot as plt
import custom_runs

print('IMPORTED!')

# Print the functions in the module
print("Fortran functions in the module:")
for thing in dir(custom_runs.custom_runs_module):
    if not thing.startswith('__'):
        print( '\t - ', thing)

collision_param = [-9.93853,-4.5805,3.27377,-0.50008,-2.45565,-1.07799,23.33004,24.69427,3.49825,5.32056,309.6923,36.8125,41.78471,51.42857,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.94594,0,0,0.18901,-22.47999,9.02372,0.0,1.0,0.0,0.0,0.0,0.0][0:22]

collision_param =np.array(collision_param).astype(np.float64)

print( collision_param.shape )

# Convert the array to Fortran contiguous
collision_param = np.asfortranarray(collision_param)

# Define other parameters
npts1 = 4000
npts2 = 4000
heat1 = 0.1
heat2 = 0.2

# Call the Fortran function
print("PY: Calling Basic Disk:")
ipts = custom_runs.custom_runs_module.basic_disk( collision_param, npts1, npts2 )

print("PY: Calling basic run Once!")
ipts, fpts = custom_runs.custom_runs_module.basic_run( collision_param, npts1, npts2 )
print("PY: tmp pts: ", ipts.shape)

# plot initial particles
# size and color of points
plt.scatter(ipts[:,0], ipts[:,1], s=1, c='b', label='Fortran Module')

# plot fpts 
plt.scatter(ipts_4k[:,0], ipts_4k[:,1], s=1, c='r', label='Fortran CMD Line')
plt.legend()

# save plot
plt.savefig('tmp_init.png')

# clear figure
plt.clf()

# plots final particles
plt.scatter(fpts[:,0], fpts[:,1], s=1, c='b', label='Fortran Module')
plt.scatter(fpts_4k[:,0], fpts_4k[:,1], s=1, c='r', label='Fortran CMD Line')
plt.legend()
plt.savefig('tmp_final.png')

