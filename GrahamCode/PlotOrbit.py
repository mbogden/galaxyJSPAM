# Graham West
from copy import deepcopy
import sys 
import time 
import random
import numpy as np
import numpy.linalg as LA
import math
import pandas as pd
from subprocess import call
from scipy import optimize
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib import image as img

import pickle
import cv2
import multiprocessing as mp
import threading as thr
from threading import Thread
import os
import time
import Queue

from mpl_toolkits.mplot3d import Axes3D

import glob

from math import sin
from math import cos


##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	###############
	###############
	# it is easy to construct a system where the true rmin hasnt happened yet 
	###############
	###############
	
	pFile = "587722984435351614_combined.txt"
#	pFile = "587729227151704160_combined.txt"
	
	toPlot    = 1
	
	nBin      = 60		# bin resolution
	nParam    = 14		# number of SPAM parameters
	
	targetInd = 0
	
	plotUnp = 1
	
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	bound = [	[-10, 10],
			[-10, 10],
			[-10, 10] ]
	bound = np.array(bound)
	
	pReal = [	6, 6, 0, 
			0, 2, 0,
			1, 30,
			2, 2,
			1, 0,
			1, 0 ]
	pReal = np.array( pReal )
	
	pp = pReal[10]*np.pi/180.0
	sp = pReal[11]*np.pi/180.0
	pt = pReal[12]*np.pi/180.0
	st = pReal[13]*np.pi/180.0
	
	# get simulated target
	print "basic_run_unperturbed"
	T, V, RVt, RVv   = solve( pReal, nParam, nBin, bound[0:2,:], "00" )
	print "mod_run"
	tmin, dmin, rmin = solveMod( pReal, nParam )
	print "orb_run"
	orbit = solveOrb( pReal, nParam )
	
	
	####################
	###   ANALYSIS   ###
	####################
	
	nLines, xxx = RVt.shape
	nPts = nLines - 1
	
	
	####################
	###   PLOTTING   ###
	####################
	
	labels = [ 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mr', 'mt', 'rp', 'rs', 'pp', 'ps', 'tp', 'ts' ]
	
	print "plotting"
	if(   toPlot == 1 ):
		fig = plt.figure()
		fig.set_size_inches(11,11)
		ax = Axes3D(fig)
		
		if( plotUnp == 1 ):
			# plot prim
			ax.plot( RVv[:nPts/2,0], RVv[:nPts/2,1], RVv[:nPts/2,2], 'k.'  )
			# plot sec
			ax.plot( RVv[nPts/2:,0], RVv[nPts/2:,1], RVv[nPts/2:,2], 'k.'  )
		else:
			# plot prim
			ax.plot( RVt[:nPts/2,0], RVt[:nPts/2,1], RVt[:nPts/2,2], 'k.'  )
			# plot sec
			ax.plot( RVt[nPts/2:,0], RVt[nPts/2:,1], RVt[nPts/2:,2], 'k.'  )
		# end
		
		ax.plot( orbit[:,0], orbit[:,1], orbit[:,2], 'm' )
		
		"""
		# plot prim center
#		ax.plot( [ origin[0] ], [ origin[1] ], [ origin[2] ], 'kX' )
		# plot sec center
#		ax.plot( [ secCen[0] ], [ secCen[1] ], [ secCen[2] ], 'gX'  )
		
		# plot velocity vector
		ax.plot( [ secCen[0], secCen[0]+secVn[0]*pReal[9]*4], [ secCen[1], secCen[1]+secVn[1]*pReal[9]*4], [ secCen[2], secCen[2]+secVn[2]*pReal[9]*4], 'g', linewidth=3  )
		
		# plot rmin
		ax.plot( [ 0, rmin[0]], [ 0, rmin[1]], [ 0, rmin[2]], 'm', linewidth=3  )
		
		# plot prim orientation
		ax.plot( [ 0, pVec2[0]], [ 0, pVec2[1]], [ 0, pVec2[2]], 'r', linewidth=3  )
		# plot sec orientation
		ax.plot( [ secCen[0], secCen[0]+sVec2[0]], [ secCen[1], secCen[1]+sVec2[1]], [ secCen[2], secCen[2]+sVec2[2]], 'r', linewidth=3  )
		
		# plot orbit orientation
		ax.plot( [ 0, oVec2[0]], [ 0, oVec2[1]], [ 0, oVec2[2]], 'g', linewidth=3  )
		
		# plot ascending node
		ax.plot( [ 0, ascNode2[0]], [ 0, ascNode2[1]], [ 0, ascNode2[2]], 'b', linewidth=3  )
		
		# plot reference direction
		ax.plot( [ 0, refDir2[0]], [ 0, refDir2[1]], [ 0, refDir2[2]], 'y', linewidth=3  )
		
		# plot orbit
#		ax.plot( orbit[:,0], orbit[:,1], orbit[:,2], 'k', linewidth=2  )
		"""
		
		ax.set_xlim( bound[0] )
		ax.set_ylim( bound[1] )
		ax.set_zlim( bound[2] )
			
		ax.set_xlabel( 'x' )
		ax.set_ylabel( 'y' )
		ax.set_zlabel( 'z' )
	# end	
	
	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.show()
	
	"""
	bound = np.array([	# bin window
#		[-0.6,0.6],
#		[-0.4,0.8]])
		[-19.0, 11.0],
		[-18.0, 12.0]])
	"""
	
	"""
	# read zoo file
	data, nModel, nCol = ReadAndCleanupData( pFile )
	
	pReal = data[targetInd,0:-1]
	
	# get parameter stats
	mins = np.min( data, axis=0 )[0:-1]
	maxs = np.max( data, axis=0 )[0:-1]
#	stds = np.std( data, axis=0 )[0:-1]
	"""
	
##############
#  END MAIN  #
##############

def getOrbitalElements( pReal, RVt, RVv, rmin ):
	
	pp = pReal[10]*np.pi/180.0
	sp = pReal[11]*np.pi/180.0
	pt = pReal[12]*np.pi/180.0
	st = pReal[13]*np.pi/180.0
	
	origin = np.zeros(3)
	
	secCen = RVt[-1,:3]
	secVel = RVt[-1,3:]
	secVm  = ( secVel[0]**2 + secVel[1]**2 + secVel[2]**2 )**0.5
	secVn  = secVel/secVm
	
	pVec   = np.array( [ sin(pt)*cos(pp), sin(pt)*sin(pp), cos(pt) ] )
	
	sVec   = np.array( [ sin(st)*cos(sp), sin(st)*sin(sp), cos(st) ] )
	
	oVec   = np.cross( secCen, secVel )
	oVec   = oVec/LA.norm(oVec)
	
	inc = math.acos( np.dot( oVec, pVec ) )/np.pi*180.0
	
	ascNode = np.cross( pVec, oVec )
	ascNode = ascNode/LA.norm(ascNode)
	
	argPer = math.acos( np.dot( ascNode, rmin[:3]/LA.norm(rmin[:3]) ) )*180.0/np.pi
	
	refDir  = np.array( [ cos(pp), sin(pp), 0 ] )
	
	xxx = np.cross( refDir, ascNode )
	yyy = np.dot( pVec, xxx )
	if( yyy >= 0 ):
		longAN = math.acos( np.dot( refDir, ascNode ) )*180.0/np.pi
	else:
		longAN = 360 - math.acos( np.dot( refDir, ascNode ) )*180.0/np.pi
	# end
	
	return inc, argPer, longAN
# end

def ReadAndCleanupData( filePath ):
	
	print "Cleaning target file..."
	
	# read data into np array
	df = pd.read_csv( filePath, sep=',|\t', engine='python', header=None )
	data1 = df.values
	
	# remove unranked models
	ind = 0
	while( not math.isnan(data1[ind,-1]) ):
		ind += 1
	# end
	data2 = data1[0:ind,:]
	nModel = ind + 1
	
	# include human score and SPAM params
	cols = range(4,18) + [ 1 ]
	data2 = data2[:,cols]
	
	# convert p,s mass to ratio,total mass
	r = data2[:,6] / data2[:,7]
	t = data2[:,6] + data2[:,7]
	data2[:,6] = r
	data2[:,7] = t
	
	data2 = np.array( data2, dtype=np.float32 )
	
	return data2, nModel, len(cols)
	
# end

def solve( param, nParam, nBin, bound, fileInd ):
	
	p = deepcopy(param)
	
	# convert mass units
	r    = p[6]
	t    = p[7]
	p[7] = t/(r+1)
	p[6] = r*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	# old idk	
#	call("./basic_run " + paramStr + "", shell=True)
	# no flag
#	call("./basic_run_unpreturbed " + paramStr + " > SolveMetro.out", shell=True)
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
	# no flag
#	RV   = np.loadtxt("a.101")
#	RV_u = np.loadtxt("a.000")
	
	# with flag
	RV   = np.loadtxt("basic_"     + fileInd + ".out")
	RV_u = np.loadtxt("basic_unp_" + fileInd + ".out")
#	print RV_u[0,:]
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
	binC   = BinField( nBin, RV,   bound )
	binC_u = BinField( nBin, RV_u, bound )
	
	return binC, binC_u, RV, RV_u
	
# end

def solveMod( param, nParam ):
	
	p = deepcopy(param)
	
	# convert mass units
	r    = p[6]
	t    = p[7]
	p[7] = t/(r+1)
	p[6] = r*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	call("./mod_run " + paramStr + " > SolveMetro.out", shell=True)
	
	output = np.loadtxt("rmin.txt")
	
	tmin = output[0]
	dmin = output[1]
	rmin = output[2:]
	
	return tmin, dmin, rmin
	
# end

def solveOrb( param, nParam ):
	
	p = deepcopy(param)
	
	# convert mass units
	r    = p[6]
	t    = p[7]
	p[7] = t/(r+1)
	p[6] = r*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	call("./orb_run " + paramStr + " > SolveMetro.out", shell=True)
	
	orbit = np.loadtxt( "orbit.txt" )
	
	return orbit
	
# end

def BinField( nBin, RV, bound ):
	
	nPts = np.size(RV,0)-1
#	print nPts
	
	binC  = np.zeros((nBin,nBin))
	
	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	
	for i in range(nPts):
		x = float(RV[i,0])
		y = float(RV[i,1])
		
		ii = int( (x - xmin) / (xmax - xmin) * nBin )
		jj = int( (y - ymin) / (ymax - ymin) * nBin )
		
		if( ii >= 0 and ii < nBin and jj >= 0 and jj < nBin ):
		        binC[jj,ii] = binC[jj,ii] + 1
		# end
	# end
	
	return binC
	
# end








def equals(a, b):
	
	n = len(a)
	
	test = 1
	for i in range(n):
		if( a[i] != b[i] ):
			test = test*0
	
	return test
	
# end

def print2D(A):
	
	print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in A]))
	
# end






main()






