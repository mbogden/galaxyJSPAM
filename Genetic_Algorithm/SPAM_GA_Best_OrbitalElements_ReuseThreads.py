 # Graham West
from copy import deepcopy
#import sys 
#import time 
import pandas as pd
import random
import numpy as np
import numpy.linalg as LA
import math
from subprocess import call
from scipy import misc
from matplotlib import pyplot as plt
#from matplotlib import image as img

import pickle
#import cv2
#import multiprocessing as mp
import threading as thr
from threading import Thread
import os
import time
import Queue

import socket as skt

#import glob

from math import sin
from math import cos


##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	np.set_printoptions(precision=3, suppress=True)
	
	hostname  = skt.gethostname()
	toPlot    = -1
	
	# target ID
#	zooBase = "587722984435351614"
#	zooBase = "587736941981466667"
	zooBase = "587726033843585146"
	
	ext     = "_combined.txt"
	pFile   = zooBase + ext
	
	fileInd   = "test"
	
	targetInd = 0
	outBase   = "Results_GA_" + zooBase + "_" + fileInd + "_"
	
	nProc     = 2**2	# number of threads
	
	# GA params
	nPop      = 2**2	# size of population at each step
	nGen      = 2**1	# number of generations
#	nKeep     = 2**2	# elitism
	nKeep     = 0		# elitism
	nPhase    = 2
	
	# image params
	nBin      = 35		# bin resolution
	nParam    = 14		# number of SPAM parameters
	scrTM     = 8		# which machine score
	scrMU     = 8		# which machine score
	b         = 0.15	# stddev of perturbedness
	reseedPerc= 0.125	# percentage of sols to reseed each step
#	reseedPerc= 0		# percentage of sols to reseed each step
	
	# parameters to fit
#	pFit = [ 2, 5 ]
#	pFit = [ 3, 4, 5 ]
#	pFit = [ 2, 3, 4, 5 ]
#	pFit = [ 2, 3, 4, 5, 6 ]
#	pFit = [ 2, 3 ]
#	pFit = [ 2, 3, 4, 5, 6, 12, 13 ]
#	pFit = [ 2, 5, 8, 12, 13 ]
#	pFit = [ 2, 5, 10, 11, 12, 13 ]
#	pFit = [ 8, 9, 10, 11, 12, 13 ]
#	pFit = [ 4, 5, 10, 11, 12, 13 ]
#	pFit = [ 8, 9, 10, 11, 12, 13 ]
#	pFit = [ 5, 6, 12, 13 ]
	
	pFit = [ 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 ]
	
	shrink   = 1.0		# fraction of max xLim
	
#	psi_p = -1
#	psi_s = -1
#	psi = [ psi_p, psi_s ]
	
	zooThresh    = 0.5	# fit thresh for zoo file
	
	# MCMC params
	sigScale = 0.05		# scale param stddev for prop width
	toMix    = 1
	mixProb  = np.array( [ 1.0, 1.0, 1.0 ] )
	mixAmp   = [ 1.0/3.0, 3.0 ]
	burn     = nGen/2**-1
	beta     = 0.001
	
	initSeed = 234329	# random seed for initial params
#	np.random.seed(initSeed)
#	random.seed(initSeed)
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	# read zoo file
	data, psi, nModel, nCol = ReadAndCleanupData( pFile, zooThresh )
	print "num zoo models: " + str(nModel)
	pReal   = data[targetInd,0:-1]
	psiReal = psi[targetInd,:]
	
	# read max xLim and window bounds
	XXX = np.loadtxt( "Bounds_" + zooBase + ".txt" )
#	mmm = XXX[:nParam,:]
	bound = XXX[nParam:,:]
	
	# get parameter stats
	mins = np.min( data, axis=0 )[0:-1]
	maxs = np.max( data, axis=0 )[0:-1]
	mmm = [ mins, maxs ]
	mmm = np.transpose(np.array(mmm))
#	print mmm
	# modify max xLim
#	mmm[2,:]  = np.array([ -10.0,  10.0 ])
#	mmm[2,:]  = np.array([   0.0,   5.0 ])
	mmm[2,:]  = np.array([   0.0,   mmm[2,1] ])  # sym
	"""
	mmm[3,:]  = np.array([  -0.9,   0.9 ])
	mmm[4,:]  = np.array([   0.0, 360.0 ])
	mmm[5,:]  = np.array([ -90.0,  90.0 ])
#	mmm[6,:]  = np.array([   0.2,   0.8 ])
#	mmm[7,:]  = np.array([  10.0,  70.0 ])
#	mmm[8,:]  = np.array([   0.5,  10.0 ])
#	mmm[9,:]  = np.array([   0.5,  10.0 ])
	"""
	mmm[10,:] = np.array([   0.0, 180.0 ])	# sym
	mmm[11,:] = np.array([   0.0, 180.0 ])	# sym
	mmm[12,:] = np.array([ -89.0,  89.0 ])	
	mmm[13,:] = np.array([ -89.0,  89.0 ])	
	# shrink xLim
	xLim = np.zeros((nParam,2))
	for i in range(nParam):
		xLim[i,0] = pReal[i] - shrink*(pReal[i] - mmm[i,0])
		xLim[i,1] = shrink*(mmm[i,1] - pReal[i]) + pReal[i]
	# end
	
	print "    min      real     max"
	for i in range(nParam):
#		print xLim[i,0], pReal[i], xLim[i,1]#, pWidth[i]
		print( "{:8.2f} ".format(xLim[i,0]) + "{:8.2f} ".format(pReal[i]) + "{:8.2f} ".format(xLim[i,1]) )
	# end
	print " "
	
	# get simulated target
	T, V, RVt, RVv = solve( pReal, nParam, nBin, bound, "00", psiReal )
	nPts, xxx = RVt.shape
	
	# find target perturbedness
	muScore = MachineScore( nBin, T, V, scrMU )
	a = muScore
	
	
	##############
	###   GA   ###
	##############
	
	# create job q and out q
	qJob = TaskQueue(num_workers=nProc)
	qOut = Queue.Queue()
	
	# RUN GA, true target
	chain, scores = GA( nProc, nGen, nPop, nParam, pFit, pReal, xLim, nBin, bound, T, RVt, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b, hostname, nKeep, sigScale, psiReal, reseedPerc, qJob, qOut )
	
	if( nPhase > 0 ):
		nGen = nGen/nPhase
	# end
	
	for phase in range(nPhase):
		
		# get best solution from chain1
		maxScoInd = np.unravel_index( np.argmax(scores, axis=None), scores.shape )
		pBest = chain[maxScoInd[0],maxScoInd[1],:]
		
		if( phase % 2 == 0 ):
			pFit = [ 4, 5, 10, 11, 12, 13 ]
		else:
			pFit = [ 2, 3, 6, 8, 9 ]
		# end
		
		# RUN GA, previous best model as target
		chain1, scores1 = GA( nProc, nGen, nPop, nParam, pFit, pBest, xLim, nBin, bound, T, RVt, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b, hostname, nKeep, sigScale, psiReal, reseedPerc, qJob, qOut )
	
		chain  = np.concatenate( ( chain,  chain1  ) )
		scores = np.concatenate( ( scores, scores1 ) )
	# end
	
	# write GA output to files
	pickle.dump( chain,   open( outBase + "solutions.txt", "wb" ) )
	pickle.dump( scores,  open( outBase + "scores.txt"   , "wb" ) )
#	pickle.dump(      M,  open( outBase + "models.txt"   , "wb" ) )
	
	
	##############################################
	# calculate orbital elemnts for GA solutions #
	##############################################
	nGen = chain.shape[0]
	
	orbitals = OrbitalGA( nProc, nGen, nPop, nParam, chain )
	
	# write orbital elemts
	pickle.dump( orbitals,   open( outBase + "elements.txt", "wb") )
	
	
	####################
	###   ANALYSIS   ###
	####################
	
	nGen, nPop, nParam = chain.shape
	
	indBest = np.unravel_index( np.argmax(scores, axis=None), scores.shape )
	pBest   = chain[indBest[0],indBest[1],:]
	
	print pReal
	print pBest
	print scores[indBest[0],indBest[1]]
	
	print outBase
	
#	M, U, RVm, RVu = solve( pBest, nParam, nBin, bound, "00" )
	
	
	####################
	###   PLOTTING   ###
	####################
	
	labels = [ 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mf', 'mt', 'rp', 'rs', 'pp', 'ps', 'tp', 'ts' ]
	
	if(   toPlot == 1 ):
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		ind = 0
		for i in range(nParam+1):
			if(   ind < nParam ):
				axes[ind].plot( np.sort( chain[:,:,ind], axis=1), 'b.' )
				axes[ind].plot(  xLim[ind,0]*np.ones(nGen+1), 'g-' )
				axes[ind].plot(  xLim[ind,1]*np.ones(nGen+1), 'g-' )
				axes[ind].plot( pReal[ind]*np.ones(nGen+1), 'r-' )
				axes[ind].plot( pBest[ind]*np.ones(nGen+1), 'k-' )
				axes[ind].set_ylabel( labels[ind] )
			elif( ind == nParam ):
				axes[ind].plot( np.sort( scores, axis=1), 'r.' )
				axes[i].set_ylim( [0,       1      ] )
			# end
			ind += 1
		# end
	elif( toPlot == 2 ):
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		for i in range(nParam):
			axes[i].plot( chain[:,:,i].flatten(), scores.flatten(), 'b.' )
			
			axes[i].plot( [ pReal[i], pReal[i] ], [0, 1], 'r--' )
			
			axes[i].set_xlim( [xLim[i,0], xLim[i,1]] )
			axes[i].set_ylim( [0,       1      ] )
			axes[i].set_xlabel( labels[i] )
		# end
	elif( toPlot == 3 ):
		fig, axes = plt.subplots(nrows=3, ncols=3)
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		axes[0,0].imshow( T,   interpolation="none", cmap="gray" )
		axes[0,0].set_title("T")
		axes[0,1].imshow( M,   interpolation="none", cmap="gray" )
		axes[0,1].set_title("M")
		axes[0,2].imshow( T-M, interpolation="none", cmap="bwr" )
		axes[0,2].set_title("T-M")
		
		axes[1,0].imshow( np.log(1+T), interpolation="none", cmap="gray" )
		axes[1,0].set_title("T")
		axes[1,1].imshow( np.log(1+M), interpolation="none", cmap="gray" )
		axes[1,1].set_title("M")
		axes[1,2].imshow( np.log(T)-np.log(M), interpolation="none", cmap="bwr" )
		axes[1,2].set_title("T-M")
		
		h      = 0
		T[T>h] = 1
		M[M>h] = 1
		
		axes[2,0].imshow( T,   interpolation="none", cmap="gray" )
		axes[2,0].set_title("T")
		axes[2,1].imshow( M,   interpolation="none", cmap="gray" )
		axes[2,1].set_title("M")
		axes[2,2].imshow( T-M, interpolation="none", cmap="bwr" )
		axes[2,2].set_title("T-M")
	elif( toPlot == 4 ):
		fig, axes = plt.subplots(nrows=1, ncols=2)
		axes = axes.flatten()
		fig.set_size_inches(9,6)
		
		axes[0].plot( RVt[0:nPts/2,0],  RVt[0:nPts/2,1],  'b.' )
		axes[0].plot( RVt[nPts/2:-1,0], RVt[nPts/2:-1,1], 'r.' )
		
		axes[1].plot( RVv[0:nPts/2,0],  RVv[0:nPts/2,1],  'b.' )
		axes[1].plot( RVv[nPts/2:-1,0], RVv[nPts/2:-1,1], 'r.' )
	elif( toPlot == 5 ):
		plt.imshow( X, interpolation="none", cmap="gray" )
	elif( toPlot == 11 ):
#		fig, axes = plt.subplots(nrows=4, ncols=4)
		fig, axes = plt.subplots(nrows=3, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		gens = []
		for i in range(nGen):
			gens.append( i*np.ones(nPop) )
		# end
		gens = np.array(gens)
		
		print gens.shape
		print chain[:,:,0].shape
		
		ind = 0
#		for i in range(nParam+1):
		for i in [2,3,4,5,6,7,8,9,12,13,14]:
			if(   i < nParam ):
#				axes[ind].plot( chain[:,:,i], 'b.' )
				II = axes[ind].scatter( gens.flatten(), chain[:,:,i].flatten(), c=scores.flatten(), s=8, cmap='jet' )
				
				II.set_clim( [ 0.0, 1.0 ] )
				
				axes[ind].plot( pReal[i]*np.ones(nGen+1), 'r-', linewidth=2 )
				axes[ind].plot( pBest[i]*np.ones(nGen+1), 'g--', linewidth=3 )
				axes[ind].set_ylabel( labels[i] )
				axes[ind].set_xlabel( "generation" )
			elif( i == nParam ):
				II = axes[ind].scatter( gens.flatten(), scores.flatten(), c=scores.flatten(), s=8, cmap='jet' )
				II.set_clim( [ 0.0, 1.0 ] )
				
				axes[ind].set_ylim( [0,       1      ] )
				axes[ind].set_ylabel( "scores" )
				axes[ind].set_xlabel( "generation" )
			# end
			ind += 1
		# end

	# end	
	
	if( toPlot > 0 ):
		plt.tight_layout(w_pad=0.0, h_pad=0.0)
		plt.show()
	# end
	
	
##############
#  END MAIN  #
##############

def perturb( pm, pt ):
	if( pm <= pt ):
		r = pm/pt
	else:
		r = (1.0-pm)/(1.0-pt)
	# end
	return r
# end

def rotMat( axis, theta ):
	theta = theta * np.pi / 180.0
	axis = axis/LA.norm(axis)
	a = math.cos( theta / 2.0 )
	b, c, d = -axis * math.sin( theta / 2.0 )
	aa, bb, cc, dd = a**2, b**2, c**2, d**2
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	
	M = [	[ aa+bb-cc-dd, 2*(bc+ad), 2*(bc-ad) ],
		[ 2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab) ],
		[ 2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc ] ]
	
	return np.array(M)
# end

def sigm( x ):
	return 1.0 / ( 1.0 + np.exp(-x) )
# end

def relax( x, a ):
	return min( x/a, 1 )
# end

def relax2( dx, h ):
	if( abs(dx) < h ):
		return 0
	else:
		return abs(dx)-h
	# end
# end

def MachineScore( nBin, binCt, binCm, scr ):
	
	T = deepcopy(binCt)
	M = deepcopy(binCm)
	
	if( scr == 0 ):
		"""
		mm = max( 1.0*np.amax(binCt), 1.0*np.amax(M) )
		
		tmScore = 0.0
		for i in range(nBin):
			for j in range(nBin):
				tmScore += ( T[i,j] - M[i,j] )**2
			# end
		# end
		tmScore /= 1.0*nBin**2
		tmScore /= 1.0*mm**2
		tmScore = ( 1 - tmScore**0.5 )**2
		"""
		
		X = binCt/np.amax(binCt)
		Y = binCm/np.amax(binCm)
		
		X = np.log(1+X)
		Y = np.log(1+Y)
		
		tmScore = 0.0
		for i in range(nBin):
			for j in range(nBin):
				tmScore += ( X[i,j] - Y[i,j] )**2
			# end
		# end
		
#		s = 4
		s = 3
		tmScore = np.exp( - tmScore/(2*s**2) )
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
		if( tmScore < 0 ):
			tmScore = 0.0
		# end
	elif( scr == 4 ):
		h = 0
		
		T  = T.flatten()
		M  = M.flatten()
		
		T[T>h] = 1
		M[M>h] = 1
		
		tmScore = np.corrcoef( T, M )[0,1]
		if( tmScore < 0 ):
			tmScore = 0.0
		# end
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
	elif( scr == 8 ):
		T = T.flatten()
		M = M.flatten()
		
		tmScore = np.corrcoef( np.log(1+T), np.log(1+M) )[0,1]
		
		if( tmScore < 0 ):
			tmScore = 0.0
		# end
	# end
	
#	tmScore *= tmScore
	
	return tmScore
	
# end

def MachineScoreW( nBin, binCt, binCm, binCu, scr ):
	
	T = deepcopy(binCt)
	M = deepcopy(binCm)
	U = deepcopy(binCu)
	
	T = np.log( 1+T.flatten() )
	M = np.log( 1+M.flatten() )
	U = np.log( 1+U.flatten() )
	
	tm = np.abs(T-M)
	mu = np.abs(M-U)
	tu = np.abs(T-U)
	weights = ( tm + mu + tu ) + np.ones(len(tu))*np.mean(tm+mu+tu)
	
#	a = np.corrcoef( np.log(1+T), np.log(1+M) )[0,1]
	b = corrW( T, M, weights )
	
	return b
# end

def covW( x, y, w ):
	n = len(x)
	return np.sum( w * (x - np.average(x,weights=w)) * (y - np.average(y,weights=w)) )/np.sum(w)*n/(n-1)
# end

def corrW( x, y, w ):
	return covW(x,y,w)/( covW(x,x,w)*covW(y,y,w) )**0.5
# end

def ReadAndCleanupData( filePath, thresh ):
	
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
	
	# ignore bad zoo models
	data2 = data2[data2[:,-1]>=thresh,:]
	nModel = data2.shape[0]
	
	data2 = np.array( data2, dtype=np.float32 )
	
#	data2[0,2]  = 10
#	data2[0,5]  = -10
#	data2[0,10] = 20
#	data2[0,11] = 20
#	data2[0,12] = 100
#	data2[0,13] = 100
	
	data3 = deepcopy(data2)
	
	psi = []
	for i in range(nModel):
		psi_p = 1
		psi_s = 1
		
		if( data2[i,2] < 0 ):
			data2[i,2]  =  -1 * data2[i,2]
			data2[i,5]  =  -1 * data2[i,5]
			data2[i,10] = 180 + data2[i,10]
			data2[i,11] = 180 + data2[i,11]
		# end
		data2[i,10] %= 360
		data2[i,11] %= 360
		if( data2[i,10] > 180 ):
			data2[i,10] = data2[i,10] - 180
			data2[i,12] = -1 * data2[i,12]
		# end
		if( data2[i,11] > 180 ):
			data2[i,11] = data2[i,11] - 180
			data2[i,13] = -1 * data2[i,13]
		# end
		data2[i,12] %= 360
		data2[i,13] %= 360
		if( data2[i,12] > 180 ):
			data2[i,12] = data2[i,12] - 360
		# end
		if( data2[i,13] > 180 ):
			data2[i,13] = data2[i,13] - 360
		# end
		
		if( data2[i,12] > 90 ):
			data2[i,12] = data2[i,12] - 180
			psi_p = -1
		elif( data2[i,12] < -90 ):
			data2[i,12] = data2[i,12] + 180
			psi_p = -1
		# end
		if( data2[i,13] > 90 ):
			data2[i,13] = data2[i,13] - 180
			psi_s = -1
		elif( data2[i,13] < -90 ):
			data2[i,13] = data2[i,13] + 180
			psi_s = -1
		# end
		psi.append( [psi_p,psi_s] )
	# end
	psi = np.array(psi)
	
	# energy
	G = 1
	r = ( data2[:,0]**2 + data2[:,1]**2 + data2[:,2]**2 )**0.5
	U = -G*data2[:,6]*data2[:,7]/r
	v = ( data2[:,3]**2 + data2[:,4]**2 + data2[:,5]**2 )**0.5
	K = 0.5*data2[:,7]*v**2
#	c = np.log(1-K/U)
	c = (K+U)/(K-U)
	
	# convert p,s mass to ratio,total mass
	t = data2[:,6] + data2[:,7]
	f = data2[:,6] / t
	data2[:,6] = f
	data2[:,7] = t
	
	# spherical velocity
	phi   = ( np.arctan2( data2[:,4], data2[:,3] ) * 180.0 / np.pi ) % 360
	theta = ( np.arcsin( data2[:,5] / v ) * 180.0 / np.pi )
	
#	data2[:,2] = c
#	data2[:,3] = v
	data2[:,3] = c
#	data2[:,2] = K
#	data2[:,3] = U
	
	data2[:,4] = phi
	data2[:,5] = theta
	
#	"""
	Ap = np.abs(data2[:,8]**2*np.cos(data2[:,12]*np.pi/180.0))
	As = np.abs(data2[:,9]**2*np.cos(data2[:,13]*np.pi/180.0))
	
	data2[:,8] = Ap
	data2[:,9] = As
#	"""
	
	return data2, psi, nModel, len(cols)
	
# end

def solve( param, nParam, nBin, bound, fileInd, psi ):
	
	p = deepcopy(param)
#	print p
	
	# add for psi
	if( psi[0] == -1 ):
		p[12] += 180.0
	# end
	if( psi[1] == -1 ):
		p[13] += 180.0
	# end
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	# K/U
#	v = ( 2*G*p[6]*(np.exp(c)-1)/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	# K+U/K-U
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	"""
	c = p[2]
	v = p[3]
	z = ( ( 2*G*p[6]*(np.exp(c)-1)/p[3]**2 )**2 - p[0]**2 - p[1]**2 )**0.5
	if( np.isreal(z) ):
		p[2] = z
	else:
		p[2] = 0
	# end
	"""
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
#	"""
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
#	"""
	
#	p[2] = 1000
	
#	print p
	paramStr = ','.join( map(str, p[0:nParam]) ) + ',0'
#	print paramStr
	
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
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

def solve_parallel( XXX, nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut, psi ):
	
	index = XXX[0]
	param = XXX[1]
	
	p = deepcopy(param)
#	print p
	
	# add for psi
	if( psi[0] == -1 ):
		p[12] += 180.0
	# end
	if( psi[1] == -1 ):
		p[13] += 180.0
	# end
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	# K/U
#	v = ( 2*G*p[6]*(np.exp(c)-1)/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	# K+U/K-U
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	"""
	c = p[2]
	v = p[3]
	z = ( ( 2*G*p[6]*(np.exp(c)-1)/p[3]**2 )**2 - p[0]**2 - p[1]**2 )**0.5
	if( np.isreal(z) ):
		p[2] = z
	else:
		p[2] = 0
	# end
	"""
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
#	"""
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
#	"""
	
#	p[2] = 1000
	
#	print p
	paramStr = ','.join( map(str, p[0:nParam]) ) + ',0'
#	print paramStr
	
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
	# with flag
	RV   = np.loadtxt("basic_"     + fileInd + ".out")
	RV_u = np.loadtxt("basic_unp_" + fileInd + ".out")
#	print RV_u[0,:]
	
	call("rm " + "basic_"     + fileInd + ".out", shell=True)
	call("rm " + "basic_unp_" + fileInd + ".out", shell=True)
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
#	M = BinField( nBin, RV,   bound )
#	U = BinField( nBin, RV_u, bound )
	
#	tmScore  = MachineScore( nBin, T, M, scrTM )
#	muScore  = MachineScore( nBin, M, U, scrMU )
#	muScoreX = np.exp( -(muScore - a)**2/(2*b**2))
#	score    = (tmScore*muScoreX)**(1.0/2.0)
	
	qOut.put( [index, RV, RV_u] )
	
#	return M, U
#	return score
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

def getInitPop( nPop, nParam, pFit, pReal, xLim ):
	
	initType = 1
	
	popSol = []
	
	if( initType == 0 ):
		for i in range(nPop):
			x = np.zeros(nParam)
			for j in range(nParam):
				if j in pFit:
					x[j] = np.random.uniform( xLim[j,0], xLim[j,1] )
				else:
					x[j] = pReal[j]
				# end
			# end
			popSol.append( x )
		# end
	else:
		maxCorr = 0.0
		for i in range(1000):
			R = []
			for j in range(nParam):
				abc = np.linspace( xLim[j,0], xLim[j,1], nPop )
				random.shuffle(abc)
				R.append( abc )
			# end
			R = np.array(R)
			
			corr = LA.det(np.corrcoef(R[pFit,:]))
			if( corr > maxCorr or i == 0 ):
				maxCorr = corr
				maxR = deepcopy(R)
			# end
#			print maxCorr
		# end
		
		for i in range(nPop):
			x = np.zeros(nParam)
			for j in range(nParam):
				if j in pFit:
					x[j] = maxR[j,i]
				else:
					x[j] = pReal[j]
				# end
			# end
			popSol.append( x )
		# end
	# end
	
	popSol = np.array(popSol)
	
	return popSol
# end

def evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, RVt, scrTM, scrMU, a, b, hostname, psi, qJob, qOut ):
	
	for j in range(nPop):
		if(   j < 10 ):
			fileInd = "00" + str(j)
		elif( j < 100 ):
			fileInd =  "0" + str(j)
		else:
			fileInd =        str(j)
		# end
		
		fileInd += "_" + hostname
#		print fileInd
		
		qJob.add_task( solve_parallel, [j, popSol[j,:]], nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut, psi )
	# end
	
	qJob.join()
	
	out  = []
	inds = []
	for i in range(nPop):
		out.append( qOut.get() )
		inds.append( out[i][0] )
	# end
	inds = np.argsort(inds)
	
	RVm = []
	RVu = []
	for i in range(nPop):
		RVm.append( out[inds[i]][1] )
		RVu.append( out[inds[i]][2] )
	# end
	RVm = np.array(RVm)
	RVu = np.array(RVu)
	
	twin = [ [min(RVt[:,0]),max(RVt[:,0])], [min(RVt[:,1]),max(RVt[:,1])] ]
	
	popFit = []
	for i in range(nPop):
#		twin = [ [min(RVt[:,0]),max(RVt[:,0])], [min(RVt[:,1]),max(RVt[:,1])] ]
		mwin = [ [min(RVm[i,:,0]),max(RVm[i,:,0])], [min(RVm[i,:,1]),max(RVm[i,:,1])] ]
		win2 = [ [min([bound[0][0],mwin[0][0]]),max([bound[0][1],mwin[0][1]])], [min([bound[1][0],mwin[1][0]]),max([bound[1][1],mwin[1][1]])] ]
		win2 = np.array(win2)
		
		T1 = BinField( nBin, RVt,        win2  )
		M  = BinField( nBin, RVm[i,:,:], bound )
		M1 = BinField( nBin, RVm[i,:,:], win2  )
		U  = BinField( nBin, RVu[i,:,:], bound )
		U1 = BinField( nBin, RVu[i,:,:], win2  )
		
		h = 0
		T1[T1>h] = 1.0
		M1[M1>h] = 1.0
		U1[U1>h] = 1.0
		
		tmScore  = MachineScoreW( nBin,  T,  M, U, scrTM )
		muScore  = MachineScore(  nBin, M1, U1,    scrMU )
		tuScore  = MachineScore(  nBin, T1, U1,    scrMU )
		muScoreX = perturb( muScore, tuScore )
		score    = tmScore*muScoreX	# fitting full perturbed model
		if( score < 0.01 ):
			score = 0.01
		# end
		
#		print "score, tmScore, muScoreX, muScore, tuScore"
#		print score, tmScore, muScoreX, muScore, tuScore
		
		popFit.append( score )
	# end
	popFit = np.array(popFit)
	
	return popFit
# end

def Selection( nPop, popSol, popFit, nKeep ):
	
	# 0: fitness-squared proportional selection
	# 1: rank proportional selection
	selectType = 0
	
	parSol = []
	parFit = []
	
	if( selectType == 0 ):
		xxx = popFit**1
#		xxx = np.ones(nPop)
		xxx = xxx/np.sum(xxx)
		popProb = np.cumsum( xxx )
		
		for i in range(nPop-nKeep):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parSol.append( [ popSol[ind1], popSol[ind2] ] )
			parFit.append( [ popFit[ind1], popFit[ind2] ] )
		# end
		
		srt = np.argsort( popFit )
		for i in range(1,nKeep+1):
			parSol.append( [ popSol[srt[-i]], popSol[srt[-i]] ] )
			parFit.append( [ popFit[srt[-i]], popFit[srt[-i]] ] )
		# end
	else:
		# get selection probabilities
		inds = popFit.argsort() + np.ones(nPop)
		popProb = np.cumsum( inds/np.sum(inds) )
		
		for i in range(nPop):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parents.append( [ popSol[ind1], popSol[ind2] ] )
		# end

	# end
	parSol = np.array(parSol)
	parFit = np.array(parFit)
	
	return parSol, parFit
# end

def Crossover( nPop, nParam, parSol, parFit, cov, step, burn ):
	
	# 0: parameter swapping w/ PCA
	# 1: mean crossover
	"""
	if( step < burn ):
		crossType = 1
	else:
		crossType = 0
	# end
	"""
	crossType = 0
	
	popSol = np.zeros((nPop,nParam))
	
	if(   crossType == 0 ):
		if( step < burn ):
			for i in range(nPop):
				for j in range(nParam):
					r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
					r  = np.random.uniform(0,1)
					
					if( r < r0 ):
						popSol[i,j] = parSol[i,0,j]
					else:
						popSol[i,j] = parSol[i,1,j]
					# end
				# end
			# end
		else:
			# get PCA
			w, v = LA.eig(cov)
			
			# convert to PCA basis
			pcaPar = []
			for i in range(nPop):
				p1 = np.dot( v, parSol[i,0,:] )
				p2 = np.dot( v, parSol[i,0,:] )
				pcaPar.append( [p1, p2] )
			# end
			pcaPar = np.array(pcaPar)
			
			# mix in PCA basis
			for i in range(nPop):
				for j in range(nParam):
					r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
					r  = np.random.uniform(0,1)
					
					if( r < r0 ):
						popSol[i,j] = pcaPar[i,0,j]
					else:
						popSol[i,j] = pcaPar[i,1,j]
					# end
				# end
			# end
			
			# convert back to parameter basis
			vinv = LA.inv(v)
			for i in range(nPop):
				popSol[i,:] = np.dot( vinv, popSol[i,:] )
			# end
		# end
		popSol = np.array(popSol)
	elif( crossType == 1 ):
		for i in range(nPop):
			r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
			c1 = parSol[i,0,:]*r0 + parSol[i,1,:]*(1-r0)
			popSol[i,:] = c1
		# end
	# end
	
	return popSol
# end

def Mutate( step, nGen, nPop, nParam, popSol, cov, xLim, pFit, nKeep ):
	
	popSol2 = np.zeros((nPop,nParam))
	
#	abc = np.linspace( xLim[2,0], xLim[2,1], nPop )
#	random.shuffle(abc)
	
	for i in range(nPop):
		
		# perturb
		x = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]
		
		# sphere walk, prim
		phi_v1   = popSol[i,4]/180.0*np.pi
		theta_v1 = popSol[i,5]/180.0*np.pi
		
		e_r_v   = np.array( [  math.cos(phi_v1)*math.cos(theta_v1), math.sin(phi_v1)*math.cos(theta_v1), math.sin(theta_v1) ] )
		e_phi_v = np.array( [                  -math.sin(phi_v1),                  math.cos(phi_v1),              0.0 ] )
		
		a_v = np.abs( np.random.normal( 0.0, cov[4,4]**0.5 ) )
		b_v = np.random.uniform(0.0,360.0)
		
		rot_e_r_v = rotMat( e_r_v, b_v )
		e_phi_rot_v = np.dot( rot_e_r_v, e_phi_v )
		
		cross_v = np.cross( e_r_v, e_phi_rot_v )
		
		rot_cross_v = rotMat( cross_v, a_v )
		e_r_rot_v = np.dot( rot_cross_v, e_r_v )
		
		phi_v2   = np.arctan2( e_r_rot_v[1], e_r_rot_v[0] )*180.0/np.pi
		theta_v2 = np.arcsin(  e_r_rot_v[2] )*180.0/np.pi
		
		x[4] = phi_v2 % 360.0
		x[5] = theta_v2
		
		# sphere walk, prim
		phi_p1   = popSol[i,10]/180.0*np.pi
		theta_p1 = popSol[i,12]/180.0*np.pi
		
		e_r_p   = np.array( [  math.cos(phi_p1)*math.cos(theta_p1), math.sin(phi_p1)*math.cos(theta_p1), math.sin(theta_p1) ] )
		e_phi_p = np.array( [                  -math.sin(phi_p1),                  math.cos(phi_p1),              0.0 ] )
		
		a_p = np.abs( np.random.normal( 0.0, cov[10,10]**0.5 ) )
		b_p = np.random.uniform(0.0,360.0)
		
		rot_e_r_p = rotMat( e_r_p, b_p )
		e_phi_rot_p = np.dot( rot_e_r_p, e_phi_p )
		
		cross_p = np.cross( e_r_p, e_phi_rot_p )
		
		rot_cross_p = rotMat( cross_p, a_p )
		e_r_rot_p = np.dot( rot_cross_p, e_r_p )
		
		phi_p2   = np.arctan2( e_r_rot_p[1], e_r_rot_p[0] )*180.0/np.pi
		theta_p2 = np.arcsin(  e_r_rot_p[2] )*180.0/np.pi
		
		x[10] = phi_p2 % 360.0
		x[12] = theta_p2
		
		# sphere walk, sec
		phi_s1   = popSol[i,11]/180.0*np.pi
		theta_s1 = popSol[i,13]/180.0*np.pi
		
		e_r_s   = np.array( [  math.cos(phi_s1)*math.cos(theta_s1), math.sin(phi_s1)*math.cos(theta_s1), math.sin(theta_s1) ] )
		e_phi_s = np.array( [                  -math.sin(phi_s1),                  math.cos(phi_s1),              0.0 ] )
		
		a_s = np.abs( np.random.normal( 0.0, cov[11,11]**0.5 ) )
		b_s = np.random.uniform(0.0,360.0)
		
		rot_e_r_s = rotMat( e_r_s, b_s )
		e_phi_rot_s = np.dot( rot_e_r_s, e_phi_s )
		
		cross_s = np.cross( e_r_s, e_phi_rot_s )
		
		rot_cross_s = rotMat( cross_s, a_s )
		e_r_rot_s = np.dot( rot_cross_s, e_r_s )
		
		phi_s2   = np.arctan2( e_r_rot_s[1], e_r_rot_s[0] )*180.0/np.pi
		theta_s2 = np.arcsin(  e_r_rot_s[2] )*180.0/np.pi
		
		x[11] = phi_s2 % 360.0
		x[13] = theta_s2

#		print xLim[4,0], x[4], xLim[4,1]
		
		# bounds check
		eps = 0.00001
		for j in range(nParam):
			if( i >= nPop-nKeep ):
				popSol2[i,j] = popSol[i,j]
			else:
				if(   xLim[j,0] > x[j] and j in pFit ):
					popSol2[i,j] = 0.5*( xLim[j,0] + popSol[i,j] )
#					popSol2[i,j] = xLim[j,0] + eps*(xLim[j,1]-xLim[j,0])
				elif( xLim[j,1] < x[j] and j in pFit ):
					popSol2[i,j] = 0.5*( xLim[j,1] + popSol[i,j] )
#					popSol2[i,j] = xLim[j,1] - eps*(xLim[j,1]-xLim[j,0])
				else:
					popSol2[i,j] = x[j]
				# end
			# end
		# end
		
	# end
	
	return popSol2
# end

def getHaarioCov( covInit, C, mean, step, burn, nPop, nParam, chainF2, popSol, pFit ):
	
	chainF = np.array(chainF2)
	
	# get AM cov matrix
	if( step < burn ):
		C    = deepcopy(covInit)
		cov2 = deepcopy(covInit)
	elif( step == burn ):
		mean = np.mean( np.array(chainF)[nPop*burn/2:,:], axis=0 )
		C    = np.cov( np.transpose( np.array(chainF)[nPop*burn/2:,:] ) )
		
		C2   = C[np.ix_(pFit,pFit)]
		w, v = LA.eig(C2)
#		cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w[pFit]) ) )**(1.0/len(pFit))
		cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w) ) )**(1.0/len(pFit))
		cov2[0,:] = covInit[0,:]
		cov2[1,:] = covInit[1,:]
		cov2[7,:] = covInit[7,:]
		cov2[:,0] = covInit[:,0]
		cov2[:,1] = covInit[:,1]
		cov2[:,7] = covInit[:,7]
	elif( step > burn ):
		for i in range(nPop):
			gamma = 1.0/(nPop*(step+1)+i)
			dx   = popSol[i] - mean
			mean = mean + gamma*dx
			C    = C    + gamma*(np.outer(dx,dx) - C)
		# end
		
		C2   = C[np.ix_(pFit,pFit)]
		w, v = LA.eig(C2)
#		cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w[pFit]) ) )**(1.0/len(pFit))
		cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w) ) )**(1.0/len(pFit))
		cov2[0,:] = covInit[0,:]
		cov2[1,:] = covInit[1,:]
		cov2[7,:] = covInit[7,:]
		cov2[:,0] = covInit[:,0]
		cov2[:,1] = covInit[:,1]
		cov2[:,7] = covInit[:,7]
	# end
	
	return cov2, mean, C
# end

def mixCov( cov, covInit, step, burn, nPop, nParam, toMix, mixProb, mixAmp ):
	
	cov2 = deepcopy(cov)
	
	# apply mixing
	if(   toMix == 1 ):
		if(   step < burn ):
			# scale
			for i in range(nParam):
				s = np.random.uniform(0,1)
				
				# thinning
				if( s <= mixProb[0] ):
					cov2[i,i] = covInit[i,i]*mixAmp[0]**2
				# widening
				elif( s <= mixProb[0] + mixProb[1] ):
					cov2[i,i] = covInit[i,i]*mixAmp[1]**2
				# fixing
				else:
					cov2[i,i] = covInit[i,i]
				# end
			# end
		elif( step >= burn ):
			# decompose, normalize
			w, v = LA.eig(cov)
			
			# mix
			for i in range(nParam):
				s = np.random.uniform(0,1)
				
				# thinning
				if( s <= mixProb[0] ):
					w[i] = w[i]*mixAmp[0]**2
				# widening
				elif( s <= mixProb[0] + mixProb[1] ):
					w[i] = w[i]*mixAmp[1]**2
				# fixing
				else:
					w[i] = w[i]
				# end
			# end
			
			# recompose matrix
			W    = np.diag(w)
#			cov2 = np.dot( np.dot(v,W), LA.inv(v) )
			cov2 = np.dot( np.dot(v,W), np.transpose(v) )
			cov2 = cov2.real
			cov2 = 0.5*( cov2 + np.transpose(cov2) )
		# end
	# end
	
	return cov2
# end

def GA( nProc, nGen, nPop, nParam, pFit, start, xLim, nBin, bound, T, RVt, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b, hostname, nKeep, sigScale, psi, reseedPerc, qJob, qOut ):
	
	# normalize
	mixProb = mixProb/np.sum(mixProb)
	
	# std is a fraction of width
	stds = np.zeros(nParam)
	for i in range(nParam):
		if i in pFit:
			stds[i] = xLim[i,1]-xLim[i,0]
		else:
			stds[i] = 0.00001
		# end
	# end
	pWidth  = stds*sigScale
	covInit = np.diag(pWidth**2)
	cov     = np.diag(pWidth**2)
	cov2    = np.diag(pWidth**2)
	C       = np.diag(pWidth**2)
	mean    = deepcopy(start)
	
#	print "xlim:"
#	print xLim
#	print "covInit"
#	print cov
	
	# get initial population
	popSol = getInitPop( nPop, nParam, pFit, start, xLim )
	popFit = evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, RVt, scrTM, scrMU, a, b, hostname, psi, qJob, qOut )
	print "step: init"
#	print np.sort(popFit)
	print np.amax(popFit)
	
	# get best solution
	fBest   = np.max(popFit)
	bestInd = np.argmax(popFit)
	pBest   = popSol[bestInd,:]
	
	adaCount = 0
	
	# add init to all
	chain = [ popSol ]
	fit   = [ popFit ]
	
	chainF = []
	for i in range(nPop):
		chainF.append( popSol[i] )
	# end

	# Random walk
	for step in range(nGen):
		print "step: " + str(step+1) + "/" + str(nGen)
		
		# reseed bottom percentage
#		if( step in flips ):
		if( step > 0 and reseedPerc > 0):
			ind = max( 1, int(reseedPerc*nPop) )
			popSol_re = getInitPop( ind, nParam, pFit, start, xLim )
			order = np.argsort(popFit)
			popSol[order[:ind],:] = popSol_re
		# end
		
		# perform selection
		parSol, parFit = Selection( nPop, popSol, popFit, nKeep )
		
		# perform crossover
		popSol = Crossover( nPop, nParam, parSol, parFit, cov, step, nGen )
		
		# get covariance matrix
		cov, mean, C = getHaarioCov( covInit, C, mean, step, burn, nPop, nParam, chainF, popSol, pFit )
		cov2         = mixCov(  cov, covInit, step, burn, nPop, nParam, toMix, mixProb, mixAmp )
#		print np.array2string( cov, max_line_width=np.inf )
#		print LA.det(cov2)**(1.0/nParam)
		
		# perform mutation
		popSol = Mutate( step, nGen, nPop, nParam, popSol, cov2, xLim, pFit, nKeep )
		
		# calculate fits
		popFit = evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, RVt, scrTM, scrMU, a, b, hostname, psi, qJob, qOut )
		
		# get best solution
		fTest = np.max(popFit)
		if( fBest < fTest ):
			fBest   = fTest
			bestInd = np.argmax(popFit)
			pBest   = popSol[bestInd,:]
		# end
		
		chain.append( popSol )
		fit.append(   popFit )
		for i in range(nPop):
			chainF.append( popSol[i] )
		# end
		
#		print np.sort(popFit)
		print np.amax(popFit)
		print popSol[np.argmax(popFit),:]
#		print " "
	# end
	chain = np.array(chain)
	fit   = np.array(fit)
	
	print " "
	
	return chain, fit

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
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) ) + ",0"
	
	call("./mod_run " + paramStr + " > SolveMetro.out", shell=True)
	
	output = np.loadtxt("rmin.txt")
	
	tmin = output[0]
	dmin = output[1]
	rmin = output[2:]
	vmin = LA.norm(output[5:])
	
	return tmin, dmin, vmin, rmin
	
# end

def solveMod_parallel( XXX, nParam, qOut ):
	
	index = XXX[0]
	param = XXX[1]
	
	p = deepcopy(param)
	psi = [1,1]
	
	"""
	# add for psi
	if( psi[0] == -1 ):
		p[12] += 180.0
	# end
	if( psi[1] == -1 ):
		p[13] += 180.0
	# end
	"""
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
	
	paramStr = ','.join( map(str, p[0:nParam]) ) + ",0"
	call("./mod_run " + paramStr + " > SolveMetro.out", shell=True)
	output = np.loadtxt("rmin.txt")
	
	tmin = output[0]
	dmin = output[1]
	rmin = output[2:]
	vmin = LA.norm(output[5:])
	beta = (p[6]+p[7])/(vmin*dmin**2)
	
	inc, argPer, longAN = getOrbitalElements( p, rmin )
	
	qOut.put( [index, tmin, dmin, vmin, beta, inc, argPer, longAN] )
	
	return 0
# end

def popOrbitals( nProc, nPop, popSol, nParam ):
	
	qJob = TaskQueue(num_workers=nProc)
	qOut = Queue.Queue()
	
	for j in range(nPop):
		qJob.add_task( solveMod_parallel, [j, popSol[j,:]], nParam, qOut )
	# end
	qJob.join()
	
	out  = []
	inds = []
	for i in range(nPop):
		out.append( qOut.get() )
		inds.append( out[i][0] )
	# end
	inds = np.argsort(inds)
	out = np.array(out)
	
	orbitals = []
	for i in range(nPop):
		orbitals.append( out[inds[i],1:] )
	# end
	orbitals = np.array(orbitals)
	
	return orbitals
# end

####################
#### ORBITAL GA ####
####################
def OrbitalGA( nProc, nGen, nPop, nParam, chain ):
	
	orbitals = []
	for i in range(nGen):
		popSol = chain[i,:,:]
		
		orb1 = popOrbitals( nProc, nPop, popSol, nParam )
		orbitals.append( orb1 )
	# end
	orbitals = np.array(orbitals)
	
	return orbitals
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






