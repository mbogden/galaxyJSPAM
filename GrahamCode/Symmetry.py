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
	
	toPlot = 2
	
	pFile = "587722984435351614_combined.txt"
#	pFile = "587729227151704160_combined.txt"
	
	toPlot    = 2
	
	nBin      = 40		# bin resolution
	nParam    = 14		# number of SPAM parameters
	
	targetInd = 0
	
	plotUnp = 1
	
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	bound = [	[-7, 13],
			[-10, 10],
			[-10, 10] ]
	bound = np.array(bound)
	
	pReal = [	0, 5, 0, 
			4, 0, 0,
			1, 30,
			4, 4,
			0, 0,
			0, 0 ]
	pReal = np.array( pReal )
	
	pTrans = deepcopy(pReal)
	
	# x symmetry
#	pTrans[3]  = -pTrans[3]
	pTrans[12] = 180 + pTrans[12]
	pTrans[13] = 180 + pTrans[13]
#	pTrans[10] = 360 - pTrans[10]
#	pTrans[11] = 360 - pTrans[11]
	
	# z symmetry
#	pTrans[2]  = -pTrans[2]
#	pTrans[5]  = -pTrans[5]
#	pTrans[12] = 360 - pTrans[12]
#	pTrans[13] = 360 - pTrans[13]
	
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	RVm1, RVu1, MR, UR = solve( pReal,  nParam, nBin, bound )
	RVm2, RVu2, MT, UT = solve( pTrans, nParam, nBin, bound )
	
	nPts, xxx = RVm2.shape
	print nPts
	
	orb1 = solveOrb( pReal,  nParam )
	orb2 = solveOrb( pTrans, nParam )
	
	
	
	####################
	###   PLOTTING   ###
	####################
	
	labels = [ 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mr', 'mt', 'rp', 'rs', 'pp', 'ps', 'tp', 'ts', 'tmin', 'dmin', 'vmin', 'beta', 'inc', 'argPer', 'longAN']
	
	if(   toPlot == 1 ):
		fig, axes = plt.subplots(nrows=1, ncols=3)
		ax = axes.flatten()
		fig.set_size_inches(12,8)
		
		ax[0].plot( RVm1[0:nPts/2,0], RVm1[0:nPts/2,1], 'b.' )
		ax[0].plot( RVm1[nPts/2:-1,0], RVm1[nPts/2:-1,1], 'r.' )
		
		ax[1].plot( RVm2[0:nPts/2,0], RVm2[0:nPts/2,1], 'b.' )
		ax[1].plot( RVm2[nPts/2:-1,0], RVm2[nPts/2:-1,1], 'r.' )
		
		ax[2].plot( orbitR1[:,0], orbitR1[:,1], orbitR1[:,2], 'k', linewidth=5 )
		ax[2].plot( orbitR2[:,0], orbitR2[:,1], orbitR2[:,2], 'm', linewidth=5 )
		ax[2].plot( [0], [0], [0], 'kX', markersize=14 )
	elif( toPlot == 2 ):
		fig, axes = plt.subplots(nrows=1, ncols=2)
		ax = axes.flatten()
		fig.set_size_inches(12,8)
		
		ax[0].plot( RVm1[0:nPts/2,0], RVm1[0:nPts/2,1], 'b.' )
		ax[0].plot( RVm1[nPts/2:-1,0], RVm1[nPts/2:-1,1], 'r.' )
		ax[0].plot( orb1[:,0], orb1[:,1], 'g', linewidth=3 )
		ax[0].set_xlim( bound[0,:] )
		ax[0].set_ylim( bound[1,:] )
		
		ax[1].plot( RVm2[0:nPts/2,0], RVm2[0:nPts/2,1], 'b.' )
		ax[1].plot( RVm2[nPts/2:-1,0], RVm2[nPts/2:-1,1], 'r.' )
		ax[1].plot( orb2[:,0], orb2[:,1], 'g', linewidth=3 )
		ax[1].set_xlim( bound[0,:] )
		ax[1].set_ylim( bound[1,:] )
	elif( toPlot == 3 ):
		fig, ax = plt.subplots(nrows=1, ncols=2)
		ax = ax.flatten()
		fig.set_size_inches(12,8)
		
		ax[0].imshow( np.log(1+MT), cmap='gray', interpolation='none' )
		ax[0].set_title( "Real" )
		
		ax[1].imshow( np.log(1+MR), cmap='gray', interpolation='none' )
		ax[1].set_title( "Trans" )
	# end	
	
	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.show()
	
	
##############
#  END MAIN  #
##############

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

def getOrbitalElements( pReal, rmin ):
	
	pp = pReal[10]*np.pi/180.0
	sp = pReal[11]*np.pi/180.0
	pt = pReal[12]*np.pi/180.0
	st = pReal[13]*np.pi/180.0
	
	origin = np.zeros(3)
	
	secCen = pReal[0:3]
#	secCen = RVt[-1,:3]
	secVel = pReal[3:6]
#	secVel = RVt[-1,3:]
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
	
	refDir  = np.array( [ cos(pt)*cos(pp), cos(pt)*sin(pp), -sin(pt) ] )
	
	xxx = np.cross( refDir, ascNode )
	yyy = np.dot( pVec, xxx )
	if( yyy >= 0 ):
		longAN = math.acos( np.dot( refDir, ascNode ) )*180.0/np.pi
	else:
		longAN = 360 - math.acos( np.dot( refDir, ascNode ) )*180.0/np.pi
	# end
	
	return inc, argPer, longAN
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

def MachineScore( nBin, binCt, binCm, scr ):
	
	T = deepcopy(binCt)
	M = deepcopy(binCm)
	
	if( scr == 0 ):
		mm = max( 1.0*np.amax(binCt), 1.0*np.amax(M) )
		
		tmScore = 1.0
		for i in range(nBin):
			for j in range(nBin):
				tmScore += ( T[i,j] - M[i,j] )**2
			# end
		# end
		tmScore /= 1.0*nBin**2
		tmScore /= 1.0*mm**2
		tmScore = ( 1 - tmScore**0.5 )**2
	elif( scr == 1 ):
		h = 0
		
		tmScore = 0
		for i in range(nBin):
			for j in range(nBin):
#				print T[i,j], M[i,j]
				tmScore += Delta( Step( T[i,j], h ) - Step( M[i,j], h ) )
			# end
		# end
		tmScore /= 1.0*nBin**2
	elif( scr == 2 ):
		h = 0
		
		x = 0
		y = 0
		z = 0
		for i in range(nBin):
			for j in range(nBin):
				if( T[i,j] > h ):
					x += 1
				# end
				if( M[i,j] > h ):
					y += 1
				# end
				if( T[i,j] > h and M[i,j] > h ):
					z += 1
				# end
			# end
		# end
		if( z > 0 ):
			tmScore = (z / (x + y - z*1.0) )
		else:
			tmScore = 0.0
		# end
		
#		tmScore = tmScore**2.0
	elif( scr == 3 ):
		T  = T.flatten()
		M  = M.flatten()
		
		tmScore = np.corrcoef( T, M )[0,1]
	elif( scr == 4 ):
		h = 0
		
		T  = T.flatten()
		M  = M.flatten()
		
		T[T>h] = 1
		M[M>h] = 1
		
		tmScore = np.corrcoef( T, M )[0,1]
	elif( scr == 5 ):
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		
		# target moments
		MT = np.zeros(10)
		moments = cv2.moments(T)
		MT[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MT[1] = c/MT[0]
		MT[2] = d/MT[0]
#		MT[3:] = cv2.HuMoments(moments)[:,0]
		MT[3] = moments['mu20']
		MT[4] = moments['mu11']
		MT[5] = moments['mu02']
		MT[6] = moments['mu30']
		MT[7] = moments['mu21']
		MT[8] = moments['mu12']
		MT[9] = moments['mu03']
		
		# model moments
		MM = np.zeros(10)
		moments = cv2.moments(M)
		MM[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MM[1] = c/MM[0]
		MM[2] = d/MM[0]
#		MM[3:] = cv2.HuMoments(moments)[:,0]
		MM[3] = moments['mu20']
		MM[4] = moments['mu11']
		MM[5] = moments['mu02']
		MM[6] = moments['mu30']
		MM[7] = moments['mu21']
		MM[8] = moments['mu12']
		MM[9] = moments['mu03']
		
		tmScore = ( (MM[1] - MT[1])/(MT[1]))**2 + ( (MM[2] - MT[2])/(MT[2]))**2
		
		tmScore = np.exp(-tmScore**1/0.1**2)
	elif( scr == 6 ):
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		
		# target moments
		MT = np.zeros(10)
		moments = cv2.moments(T)
		MT[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MT[1] = c/MT[0]
		MT[2] = d/MT[0]
#		MT[3:] = cv2.HuMoments(moments)[:,0]
		MT[3] = moments['mu20']
		MT[4] = moments['mu11']
		MT[5] = moments['mu02']
		MT[6] = moments['mu30']
		MT[7] = moments['mu21']
		MT[8] = moments['mu12']
		MT[9] = moments['mu03']
		
		# model moments
		MM = np.zeros(10)
		moments = cv2.moments(M)
		MM[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MM[1] = c/MM[0]
		MM[2] = d/MM[0]
#		MM[3:] = cv2.HuMoments(moments)[:,0]
		MM[3] = moments['mu20']
		MM[4] = moments['mu11']
		MM[5] = moments['mu02']
		MM[6] = moments['mu30']
		MM[7] = moments['mu21']
		MM[8] = moments['mu12']
		MM[9] = moments['mu03']
		
#		tmScore = ( (MM[3] - MT[3])/(MT[3]))**2 + 2*( (MM[4] - MT[4])/(MT[4]))**2 + ((MM[5] - MT[5])/(MT[5]))**2
		
		MatM = np.array( [[ MM[3], MM[4]], [MM[4], MM[5]]] )
		MatT = np.array( [[ MT[3], MT[4]], [MT[4], MT[5]]] )
		
		tmScore = ( (LA.det(MatM) - LA.det(MatT))/LA.det(MatT) )**2
		
		tmScore = np.exp(-tmScore**1/0.4**2)
	elif( scr == 7 ):
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		
		# target moments
		MT = np.zeros(10)
		moments = cv2.moments(T)
		MT[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MT[1] = c/MT[0]
		MT[2] = d/MT[0]
#		MT[3:] = cv2.HuMoments(moments)[:,0]
		MT[3] = moments['mu20']
		MT[4] = moments['mu11']
		MT[5] = moments['mu02']
		MT[6] = moments['mu30']
		MT[7] = moments['mu21']
		MT[8] = moments['mu12']
		MT[9] = moments['mu03']
		
		# model moments
		MM = np.zeros(10)
		moments = cv2.moments(M)
		MM[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MM[1] = c/MM[0]
		MM[2] = d/MM[0]
#		MM[3:] = cv2.HuMoments(moments)[:,0]
		MM[3] = moments['mu20']
		MM[4] = moments['mu11']
		MM[5] = moments['mu02']
		MM[6] = moments['mu30']
		MM[7] = moments['mu21']
		MM[8] = moments['mu12']
		MM[9] = moments['mu03']
		
		tmScore = ( (MM[6] - MT[6])/(MT[6]))**2 + 3*( (MM[7] - MT[7])/(MT[7]))**2 + 3*((MM[8] - MT[8])/(MT[8]))**2 + ((MM[9] - MT[9])/(MT[9]))**2
		
		tmScore = np.exp(-tmScore**1/20**2)
	# end
	
	return tmScore
	
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
	
	binC   = BinField( nBin, RV,   bound )
	binC_u = BinField( nBin, RV_u, bound )
	
#	return binC, binC_u
	return RV, RV_u, binC, binC_u
	
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

def getInitPop( nPop, nParam, xLim ):
	
	# get initial population
	popSol = []
	for i in range(nPop):
		x = np.zeros(nParam)
		for j in range(nParam):
			x[j] = np.random.uniform( xLim[j,0], xLim[j,1] )
		# end
		popSol.append( x )
	# end
	popSol = np.array(popSol)
	
	return popSol
# end

def evalPop( nPop, popSol, nParam, nBin,  bound, T, scrTM, scrMU, a, b ):
	
	popFit = []
	for i in range(nPop):
#		print popSol[i]
		M, U = solve( popSol[i,:], nParam, nBin, bound )
#		M = np.ones((nBin,nBin))
#		U = np.ones((nBin,nBin))
		
		tmScore = MachineScore( nBin, T, M, scrTM )
		muScore = MachineScore( nBin, M, U, scrMU )
		
		muScore2 = np.exp( -(muScore - a)**2/(2*b**2))
		
		popFit.append( tmScore*muScore2 )
	# end
	popFit = np.array(popFit)
#	print " "
	
	return popFit
# end

def Selection( nPop, popSol, popFit ):
	
	selectType = 0
	
	parents = []
	
	if( selectType == 0 ):
		# get selection probabilities
		popProb = np.cumsum( popFit/np.sum(popFit) )
		
		for i in range(nPop/2):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parents.append( [ popSol[ind1], popSol[ind2] ] )
		# end
	else:
		w = 1
	# end
	parents = np.array(parents)
	
	return parents
# end

def Crossover( nPop, nParam, parents ):
	
	crossType = 0
	
	popSol = np.zeros((nPop,nParam))
	
	if( crossType == 0 ):
		for i in range(nPop/2):
			r  = np.random.uniform(0,1)
			c1 = parents[i,0]*r     + parents[i,1]*(1-r)
			c2 = parents[i,0]*(1-r) + parents[i,1]*r
			popSol[2*i,:]   = c1
			popSol[2*i+1,:] = c2
		# end
	else:
		w = 1
	# end
	popSol = np.array(popSol)
	
	return popSol
# end

def Mutate( nPop, nParam, popSol, cov, xLim ):
	
	popSol2 = np.zeros((nPop,nParam))
	for i in range(nPop):
		popSol2[i,:] = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]
		
		for j in range(nParam):
			if(   xLim[j,0] > popSol2[i,j] ):
				popSol2[i,j] = xLim[j,0] + np.abs(xLim[j,0] - popSol2[i,j])
			elif( xLim[j,1] < popSol2[i,j] ):
				popSol2[i,j] = xLim[j,1] - np.abs(xLim[j,1] - popSol2[i,j])
			# end
		# end
	# end
	
	return popSol2
# end

def getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nParam, chainAll, popSol, r, scaleInit, toMix, mixProb, mixAmp ):
	
	chainF = np.array(chainAll)
	
	# get AM cov matrix
	if( step < burn ):
		C = deepcopy(covInit)
	elif( step == burn ):
		mean = np.mean( np.array(chainAll)[nPop*burn/2:,:], axis=0 )
		C    = np.cov( np.transpose( np.array(chainAll)[nPop*burn/2:,:] ) )
#		mean = np.mean( np.array(chainAll)[:,:], axis=0 )
#		C    = np.cov( np.transpose( np.array(chainAll)[:,:] ) )
		
#		r    = np.prod( pWidth**2 )**(0.5/nParam)
		cov  = C + beta*covInit
	elif( step > burn ):
		for i in range(nPop):
			gamma = 1.0/(nPop*step+1+i)
			dx   = popSol[i] - mean
			mean = mean + gamma*dx
			C    = C    + gamma*(np.outer(dx,dx) - C)
		# end
		
#		r    = 1.0
		cov  = C + beta*covInit
#		r    = r*np.exp( gamma*(accProb - P) )
	# end
	
	# apply mixing
	if( toMix == 1 ):
		if( step <= burn ):
			# scale
			for i in range(nParam):
				s = np.random.uniform(0,1)
				
				# thinning
				if( s <= mixProb[0] ):
					cov[i,i] = (r*mixAmp[0])**2*covInit[i,i]
				# widening
				elif( s <= mixProb[0] + mixProb[1] ):
					cov[i,i] = (r*mixAmp[1])**2*covInit[i,i]
				# fixing
				else:
					cov[i,i] = (r)**2*covInit[i,i]
				# end
			# end
		elif( step > burn ):
			# decompose, normalize
			w, v = LA.eig(cov)
			w    = scaleInit*w/np.abs(np.prod(w))**(1.0/nParam)
			
			# mix
			for i in range(nParam):
				s = np.random.uniform(0,1)
				
				# thinning
				if( s <= mixProb[0] ):
					w[i] = (r*mixAmp[0])**2*w[i]
				# widening
				elif( s <= mixProb[0] + mixProb[1] ):
					w[i] = (r*mixAmp[1])**2*w[i]
				# fixing
				else:
					w[i] = (r)**2*w[i]
				# end
			# end
			
			# recompose matrix
			W   = np.diag(w)
#			cov = np.dot( np.dot(v,W), LA.inv(v) )
			cov = np.dot( np.dot(v,W), np.transpose(v) )
			cov = cov.real
			cov = 0.5*( cov + np.transpose(cov) )
		# end
	# end
	
	return cov, mean, C
# end

def GA( nGen, nPop, nParam, start, xLim, pWidth, nBin, bound, T, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b ):
	
	covInit = np.diag(pWidth**2)
	cov     = np.diag(pWidth**2)
	C       = np.diag(pWidth**2)
	mean    = deepcopy(start)
	
	scaleInit = np.prod( pWidth**2 )**(1.0/nParam)
	r = 1.0
	
	# all generations
	chain  = []
	chainF = []
	
	# get initial population
	popSol = getInitPop( nPop, nParam, xLim )
	popFit = evalPop( nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b )
	
	# add init to all
	chain = [ popSol ]
	error = [ popFit ]
	for i in range(nPop):
		chainF.append( popSol[i] )
	# end
	
#	print "min error gen " + str(0) + ": "
#	print min(popFit)
	
	# Random walk
	for step in range(nGen):
		print str(step+1) + "/" + str(nGen)
		
		for i in range(nPop):
#			print popSol[i]
			print popFit[i]
		# end
		
		# get covariance matrix
		cov, mean, C = getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nParam, chainF, popSol, r, scaleInit, toMix, mixProb, mixAmp )
		
		# perform selection
		parents = Selection( nPop, popSol, popFit )
		
		# perform crossover
		popSol = Crossover( nPop, nParam, parents )
		
		# perform mutation
		popSol = Mutate( nPop, nParam, popSol, cov, xLim )
		
		# calculate errors
		popFit = evalPop( nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b )
		
		chain.append( popSol )
		error.append( popFit )
		for i in range(nPop):
			chainF.append( popSol[i] )
		# end
		
#		print "min error gen " + str(step+1) + ": "
#		print min(popFit)
		
		print " "
	# end
	
	chain  = np.array(chain)
	error  = np.array(error)
	
	return chain, error

# end

"""
# perform mutation
if( step <= n/3 ):
	for i in range(nPop):
		popSol[i] = Mutate( popSol[i], cov )
	# end
else:
	for i in range(nPop):
		popSol[i] = Mutate( popSol[i], cov/(1.0+np.log(step-n/3))**1.2 )
	# end
# end
"""

def cumMin( chain1, error1 ):
	
	error2 = []
	chain2 = []
	
	curMin = error1[0] + 1
	for i in range( len(error1) ):
		if( error1[i] < curMin ):
			error2.append( error1[i]   )
			chain2.append( chain1[i,:] )
			curMin = error1[i]
		else:
			error2.append( error2[-1]   )
			chain2.append( chain2[-1] )
		# end
	# end
	error2 = np.array(error2)
	chain2 = np.array(chain2)
	
	return chain2, error2
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
	
#	print('\n'.join([' '.join(['{:4}'.format(item) for item in row]) for row in A]))
#	print('\n'.join([' '.join(['{0:.2f}'.format(item) for item in row]) for row in A]))
	print('\n'.join([' '.join(['{0:8.4f}'.format(item) for item in row]) for row in A]))
	
# end






main()






