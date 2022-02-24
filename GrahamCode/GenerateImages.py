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
	
	
	zooInd = "587722984435351614"
#	zooInd = "587729227151704160"
	
	nGen      = 100		# MCMC steps
	nBin      = 60
	nParam    = 14		# number of SPAM parameters
	
	bound = np.array([	# bin window
#		[-0.6,0.6],
#		[-0.4,0.8]])
		[-19.0, 11.0],
		[-18.0, 12.0]])	
	
	shrink = 0.4
	
	direct   = ""
	fileBase = "PerturbOrbitals_" + str(zooInd)
	fileID = "00"
	
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	filePath = zooInd + "_combined.txt"
	data, nModel, nCol = ReadAndCleanupData( filePath )
	

	# read target parameters and get target bins
	pReal = data[0,0:-1]
	RVt, RVv = solve( pReal, nParam, nBin, bound )
	nPts = RVt.shape[0] # number of lines in RV file
	nDPS = RVt.shape[1] # number of cols in RV file - 6
	
	mins = np.min( data, axis=0 )[0:-1]
	maxs = np.max( data, axis=0 )[0:-1]
	xLim = np.zeros((nParam,2))
	for i in range(nParam):
		xLim[i,0] = pReal[i] - shrink*(pReal[i] - mins[i])
		xLim[i,1] = shrink*(maxs[i] - pReal[i]) + pReal[i]
	# end
	
	filePPG   = str( (nPts-1)/2 ) + "ppg"
	
	
	######################
	###   RUN MODELS   ###
	######################
	
	pVals = np.zeros((nGen,nParam+7))
	
	# get parameter range
	for j in range(nParam):
		pVals[:,j] = np.random.uniform( xLim[j,0], xLim[j,1], nGen )
	# end
	pVals[0,:nParam] = deepcopy(pReal)
	
	# ----------------------------
	
	M = np.zeros(( nGen, nBin, nBin ))
	U = np.zeros(( nGen, nBin, nBin ))
	
	# create and save RV files
	for i in range(nGen):
		RVm, RVu = solve( pVals[i,:nParam], nParam, nBin, bound )
		M[i,:,:] = BinField( nBin, RVm, bound )
		U[i,:,:] = BinField( nBin, RVu, bound )
		
		tmin, dmin, vmin, rmin = solveMod( pVals[i,:nParam], nParam )
		inc, argPer, longAN = getOrbitalElements( pVals[i,:nParam], RVm, RVu, rmin )
		
		pVals[i,14] = tmin
		pVals[i,15] = dmin
		pVals[i,16] = vmin
		pVals[i,17] = (pVals[i,6]+pVals[i,7])/(vmin*dmin**2)
		pVals[i,18] = inc
		pVals[i,19] = argPer
		pVals[i,20] = longAN
		
		print str(i+1) + "/" + str(nGen)
		sys.stdout.flush()
	# end
	
	
	fileP = fileBase + "_P_" + filePPG + "_" + fileID + ".pk"
	pickle.dump( pVals,  open(direct + fileP, "wb") )
	
	
	fileM = fileBase + "_M_" + filePPG + "_" + fileID + ".pk"
	fileU = fileBase + "_U_" + filePPG + "_" + fileID + ".pk"
	pickle.dump( M, open(direct + fileM, "wb") )
	pickle.dump( U, open(direct + fileU, "wb") )
	
	
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
	
	return data2, nModel, len(cols)
	
# end

def evalPop( nProc, nPop, popSol, nParam, nBin, bound ):
	
	qJob = TaskQueue(num_workers=nProc)
	qOut = Queue.Queue()
	
	for j in range(nPop):
		i = j + shift
		
		if( i < 10 ):
			fileInd = "00" + str(i)
		elif( i < 100 ):
			fileInd =  "0" + str(i)
		else:
			fileInd =        str(i)
		# end
		
		qJob.add_task( solve_parallel, [j, popSol[j,:]], nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut )
	# end
	
	qJob.join()
	
	out  = []
	inds = []
	for i in range(nPop):
		out.append( qOut.get() )
		inds.append( out[i][0] )
	# end
	inds = np.argsort(inds)
	
	M = []
	U = []
	for i in range(nPop):
		M.append( out[inds[i]][1] )
		U.append( out[inds[i]][2] )
	# end
	
	return popFit, M
# end

def solve_parallel( XXX, nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut ):
	
	index = XXX[0]
	param = XXX[1]
	
	p = deepcopy(param)
	
	# convert mass units
	r    = p[6]
	t    = p[7]
	p[7] = t/(r+1)
	p[6] = r*p[7]
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
	# with flag
	RV   = np.loadtxt("basic_"     + fileInd + ".out")
	RV_u = np.loadtxt("basic_unp_" + fileInd + ".out")
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
	qOut.put( [ index, RV, RV_u ] )
	
	return M, U
# end

def solve( param, nParam, nBin, bound ):
	
	p = deepcopy(param)
	
	# convert mass units
	r    = p[6]
	t    = p[7]
	p[7] = t/(r+1)
	p[6] = r*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
#	call("./basic_run " + paramStr + "", shell=True)
	call("./basic_run_unpreturbed " + paramStr + " > SolveMetro.out", shell=True)
	
	RV   = np.loadtxt("a.101")
	RV_u = np.loadtxt("a.000")
#	print RV_u[0,:]
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
#	binC   = BinField( nBin, RV,   bound )
#	binC_u = BinField( nBin, RV_u, bound )
	
#	return binC, binC_u
	return RV, RV_u
	
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
	vmin = LA.norm(output[5:])
	
	return tmin, dmin, vmin, rmin
	
# end

class TaskQueue(Queue.Queue):
	
	def __init__(self, num_workers=1):
		Queue.Queue.__init__(self)
		self.num_workers = num_workers
		self.start_workers()
	# end
	
	def add_task(self, task, *args, **kwargs):
		args = args or ()
		kwargs = kwargs or {}
		self.put((task, args, kwargs))
	# end
	
	def start_workers(self):
		for i in range(self.num_workers):
			t = Thread(target=self.worker)
			t.daemon = True
			t.start()
		# end
	# end
	
	def worker(self):
		while True:
			item, args, kwargs = self.get()
			item(*args, **kwargs)  
			self.task_done()
		# end
	# end
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






