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
# collision_param[14] = 0.5*collision_param[8]
# collision_param[15] = 0.5*collision_param[9]
collision_param = np.array(collision_param).astype(np.float64)

print( collision_param.shape )

# Convert the array to Fortran contiguous
collision_param = np.asfortranarray(collision_param)

# Define other parameters
npts1 = 1000
npts2 = 1000
heat1 = 0.1
heat2 = 0.2

# Call the Fortran function
print("PY: Calling Basic Disk:")
ipts = custom_runs.custom_runs_module.basic_disk( np.copy(collision_param), npts1, npts2, )

# create tmp scatter
n=npts1
plt.scatter(ipts[0:n,0], ipts[0:n,1], s=1, c='b', label='Primary Disk')
plt.scatter(ipts[n+1:,0], ipts[n+1:,1], s=1, c='r', label='Secondary Disk')
plt.scatter(ipts[-1,0], ipts[-1,1], s=10, c='k', label=f'Secondary Center [-1,:] ({ipts[-1,0]},{ipts[-1,1]})')

plt.legend()
plt.savefig('pts_loc.png')

# plot initial particles
# size and color of points
plt.scatter(ipts_4k[:,0], ipts_4k[:,1], s=1, c='r', label='CMD Line')
plt.scatter(ipts[:,0], ipts[:,1], s=1, c='b', label='F2PY Module')

# plot fpts 
plt.legend()

# save plot
plt.savefig('tmp_init.png')


print("PY: Calling Orbit Run!")
nsteps = custom_runs.custom_runs_module.calc_orbit_time_steps( np.copy(collision_param) )
print(f"Calculate N steps: {nsteps}")
orbit_path = custom_runs.custom_runs_module.orbit_run( np.copy(collision_param), nsteps)
print("PY: Orbit path: ", orbit_path.shape)


print("PY: Calling basic run Once!")
ipts, fpts = custom_runs.custom_runs_module.basic_run( np.copy(collision_param), npts1, npts2, 0.0, 0.0 )
print("PY: tmp pts: ", ipts.shape)

# clear figure
plt.clf()

# plots final particles
plt.scatter(fpts_4k[:,0], fpts_4k[:,1], s=1, c='r', label='CMD Line')
plt.scatter(fpts[:,0], fpts[:,1], s=1, c='b', label='F2PY Module')
#plt.plot( orbit_path[:,0], orbit_path[:,1], c='k', label='Orbit Path')
plt.legend()
plt.savefig('tmp_final.png')
plt.clf()


# Plot final and intial with path
plt.scatter(ipts[:,0], ipts[:,1], s=1, c='r', label='Initial')
plt.scatter(fpts[:,0], fpts[:,1], s=1, c='b', label='Final')
plt.plot( orbit_path[:,0], orbit_path[:,1], c='k', label='Orbit Path')
plt.legend()
plt.title('Initial, Final, and Orbit Path')
plt.savefig('tmp_orbit.png')

# Calculate index of minium radius to origin
min_r_idx = np.argmin( np.linalg.norm(orbit_path, axis=1) )

# Function to calculate the normal vector of the orbit plane
def plane_normal(points):
    # Calculate the covariance matrix of the points
    cov_matrix = np.cov(points.T)
    # Eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    # The eigenvector with the smallest eigenvalue is the normal vector of the plane
    normal_vector = eigvecs[:, np.argmin(eigvals)]
    return normal_vector

# Calculate the normal vector of the orbit path
normal_vector = plane_normal(orbit_path)

# Convert the normal vector to spherical coordinates for setting the viewing angle
r = np.linalg.norm(normal_vector)
elev = np.degrees(np.arccos(normal_vector[2] / r))
azim = np.degrees(np.arctan2(normal_vector[1], normal_vector[0]))

# Plot and save points and orbit path in 3D
plt.clf()
fig = plt.figure(figsize=(9,9)) # make figure bigger
ax = fig.add_subplot(111, projection='3d')
ax.plot(orbit_path[:,0], orbit_path[:,1], orbit_path[:,2], c='g', label='Orbit Path')
ax.quiver(0,0,0, orbit_path[min_r_idx,0], orbit_path[min_r_idx,1], orbit_path[min_r_idx,2], color='k', label='Min Radius Vector')
ax.scatter(ipts[:,0], ipts[:,1], ipts[:,2], c='r', label='Initial')
ax.scatter(fpts[:,0], fpts[:,1], fpts[:,2], c='b', label='Final')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Orbit Path')
plt.legend()

# Adjust camera angle to be pendendicular to orbit path
ax.view_init(elev=elev, azim=azim)

plt.savefig('tmp_orbit3d.png')