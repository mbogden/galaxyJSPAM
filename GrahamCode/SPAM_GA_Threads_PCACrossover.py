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



##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	
	# overlap threshold acts weird when set high relative to particles/bin
	
	pFile = "587722984435351614_combined.txt"
#	pFile = "587729227151704160_combined.txt"
	
	toPlot    = 11
	targetInd = 0
	fileInd   = "45"
	
	nProc     = 2**3
	shift     = 0*nProc
	
	scrTM     = 8		# which machine score
	scrMU     = 8		# which machine score
	
	nGen      = 2**6	# number of generations
	nPop      = 2**5	# size of population at each step
	nBin      = 40		# bin resolution
	nParam    = 14		# number of SPAM parameters
	sigScale  = 0.01		# scale param stddev for prop width
	bound = np.array([	# bin window
#		[-0.6,0.6],
#		[-0.4,0.8]])
		[-19.0, 11.0],
		[-18.0, 12.0]])
	
#	pFit = range(2,14)
#	pFit = [ 2, 5, 12, 13 ]
#	pFit = [ 2, 5 ]
	pFit = [ 2, 3, 4, 5, 6, 12, 13 ]
#	pFit = [ 2, 5, 6, 7 ]
	
	nKeep    = 2**2
	
	shrink   = 1.0
	
	toFlip   = 1
	nFlip    = 2
	flipProb = 0.5
	
	toMix    = 1
	mixProb  = np.array( [ 1.0/3.0, 1.0/3.0, 1.0/3.0 ] )
	mixAmp   = [ 1.0/2.0, 2.0 ]
	
	burn     = nGen/2**2
	beta     = 0.001
	
	b        = 0.15
	
	initSeed = 234329	# random seed for initial params
#	np.random.seed(initSeed)
	
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	# read zoo file
	data, nModel, nCol = ReadAndCleanupData( pFile )
	pReal = data[targetInd,0:-1]
	print " "
	
	# get parameter stats
	mins = np.min( data, axis=0 )[0:-1]
	maxs = np.max( data, axis=0 )[0:-1]
#	stds = np.std( data, axis=0 )[0:-1]
	mmm = [ mins, maxs ]
	mmm = np.transpose(np.array(mmm))
	
	mmm[2,:]  = np.array([ -10.0, 10.0 ])
	mmm[3,:]  = np.array([ -5.0, 5.0 ])
	mmm[4,:]  = np.array([ -5.0, 5.0 ])
	mmm[5,:]  = np.array([ -10.0, 10.0 ])
	mmm[6,:]  = np.array([ 0.25, 0.75 ])
	mmm[7,:]  = np.array([ 20.0, 70.0 ])
	mmm[8,:]  = np.array([ 2.0, 6.0 ])
	mmm[9,:]  = np.array([ 3.0, 7.0 ])
	mmm[10,:] = np.array([ 0.0, 360.0 ])
	mmm[11,:] = np.array([ 0.0, 360.0 ])
	mmm[12,:] = np.array([ 0.0, 360.0 ])
	mmm[13,:] = np.array([ 0.0, 360.0 ])
	
	xLim = np.zeros((nParam,2))
	for i in range(nParam):
		xLim[i,0] = pReal[i] - shrink*(pReal[i] - mmm[i,0])
		xLim[i,1] = shrink*(mmm[i,1] - pReal[i]) + pReal[i]
	# end
	
	# std is a fraction of width
	stds = np.zeros(nParam)
	for i in range(nParam):
		if i in pFit:
			stds[i] = xLim[i,1]-xLim[i,0]
		else:
			stds[i] = 0.0001
		# end
	# end
	pWidth = stds*sigScale
	
	print "min                 real       max           pWidth"
	for i in range(nParam):
		print xLim[i,0], pReal[i], xLim[i,1], pWidth[i]
	# end
	print " "
	
	# get simulated target
	T, V, RVt, RVv = solve( pReal, nParam, nBin, bound, "00" )
	nPts, xxx = RVt.shape
	
	# find target perturbedness
	muScore = MachineScore( nBin, T, V, scrMU )
	a = muScore
	
	
	
	##############
	###   GA   ###
	##############
	
	# RUN GA -------------
	chain, scores, M = GA( nProc, nGen, nPop, nParam, pFit, pReal, xLim, pWidth, nBin, bound, T, nFlip, flipProb,toFlip, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b, shift, nKeep )
	
	pickle.dump( chain,   open("solutions_" + fileInd + ".txt", "wb") )
	pickle.dump( scores,  open("scores_"    + fileInd + ".txt", "wb") )
	pickle.dump(      M,  open("models_"    + fileInd + ".txt", "wb") )
	
	
	
	####################
	###   ANALYSIS   ###
	####################
	
	nGen, nPop, nParam = chain.shape
	
	indBest = np.unravel_index( np.argmax(scores, axis=None), scores.shape )
	pBest   = chain[indBest[0],indBest[1],:]
	print scores[indBest[0],indBest[1]]
	
	
	
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
		fig, axes = plt.subplots(nrows=2, ncols=3)
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
#		T, V = solve( pReal, nParam, nBin, bound, "00" )
		axes[0,0].imshow( T,           interpolation="none", cmap="gray" )
		axes[0,0].set_title("T")
		
		M, U, RVm, RVu = solve( pBest, nParam, nBin, bound, "00" )
		axes[0,1].imshow( M,           interpolation="none", cmap="gray" )
		axes[0,1].set_title("M")
		
		axes[0,2].imshow( T-M, interpolation="none", cmap="bwr" )
		axes[0,2].set_title("T-M")
		
		tmScore = MachineScore( nBin, T, M, scrTM )
		muScore = MachineScore( nBin, M, U, scrMU )
		muScoreX = np.exp( -(muScore - a)**2/(2*b**2))
		score    = (tmScore*muScoreX)**(1.0/1.0)
		print score, tmScore, muScoreX
		
		h      = 0
		T[T>h] = 1
		M[M>h] = 1
		
		axes[1,0].imshow( T, interpolation="none", cmap="gray" )
		axes[1,1].imshow( M, interpolation="none", cmap="gray" )
		
		
		axes[1,2].imshow( T-M, interpolation="none", cmap="bwr" )
		axes[1,2].set_title("T-M")
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
	
	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.show()
	
	
##############
#  END MAIN  #
##############

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
	elif( scr == 8 ):
		T = T.flatten()
		M = M.flatten()
		
		tmScore = np.corrcoef( np.log(1+T), np.log(1+M) )[0,1]
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
	t = data2[:,6] + data2[:,7]
	f = data2[:,6] / t
	data2[:,6] = f
	data2[:,7] = t
	
	data2 = np.array( data2, dtype=np.float32 )
	
	return data2, nModel, len(cols)
	
# end

def solve( param, nParam, nBin, bound, fileInd ):
	
	p = deepcopy(param)
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
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

def solve_parallel( XXX, nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut ):
	
	index = XXX[0]
	param = XXX[1]
	
	p = deepcopy(param)
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
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
	
	M = BinField( nBin, RV,   bound )
	U = BinField( nBin, RV_u, bound )
	
#	tmScore  = MachineScore( nBin, T, M, scrTM )
#	muScore  = MachineScore( nBin, M, U, scrMU )
#	muScoreX = np.exp( -(muScore - a)**2/(2*b**2))
#	score    = (tmScore*muScoreX)**(1.0/2.0)
	
	qOut.put( [index, M, U] )
	
	return M, U
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
	
	initType = 0
	
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
		R = []
		for j in range(nParam):
			R.append( np.linspace( xLim[j,0], xLim[j,1], nPop ) )
		# end
		R = np.array(R)
		
		for i in range(nPop):
			x = np.zeros(nParam)
			for j in range(nParam):
				if j in pFit:
					x[j] = R[j][i]
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

def evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b, shift ):
	
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
	
	popFit = []
	for i in range(nPop):
		tmScore  = MachineScore( nBin,    T, M[i], scrTM )
		muScore  = MachineScore( nBin, M[i], U[i], scrMU )
		muScoreX = np.exp( -(muScore - a)**2/(2*b**2) )
		score    = (tmScore*muScoreX)**(1.0/1.0)
		
#		print score, tmScore, muScoreX, muScore, a
		
		popFit.append( score )
	# end
	popFit = np.array(popFit)
	
	return popFit, M
# end

def Selection( nPop, popSol, popFit, nKeep ):
	
	# 0: fitness-squared proportional selection
	# 1: rank proportional selection
	selectType = 0
	
	parSol = []
	parFit = []
	
	if( selectType == 0 ):
		xxx = popFit**2
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
	
	popSol = np.zeros((nPop,nParam))
	
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
	
	return popSol
# end

def Mutate( step, nGen, nPop, nParam, popSol, cov, xLim, pFit, toFlip, nFlip, flipProb):
	
	popSol2 = np.zeros((nPop,nParam))
	
	for i in range(nPop):
		x = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]
		
		if( ( step % (nGen/nFlip) ) == 0 and toFlip ):
			r = np.random.uniform(0,1)
			
			if( r < flipProb ):
				if(  2 in pFit and 5 in pFit ):
					x[2] = -x[2]
					x[5] = -x[5]
#				if( 10 in pFit ):
#					x[10] = 180 + x[10]
#				if( 11 in pFit ):
#					x[11] = 180 + x[11]
				if( 12 in pFit ):
					x[12] = 360 - x[12]
				if( 13 in pFit ):
					x[13] = 360 - x[13]
				# end
			# end
		# end
		
		for j in range(nParam):
			if(   xLim[j,0] > x[j] ):
				popSol2[i,j] = 0.5*( xLim[j,0] + popSol[i,j] )
			elif( xLim[j,1] < x[j] ):
				popSol2[i,j] = 0.5*( xLim[j,1] + popSol[i,j] )
			else:
				popSol2[i,j] = x[j]
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
		if( step < burn ):
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
		elif( step >= burn ):
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

def GA( nProc, nGen, nPop, nParam, pFit, start, xLim, pWidth, nBin, bound, T, nFlip, flipProb, toFlip, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b, shift, nKeep ):
	
	covInit = np.diag(pWidth**2)
	cov     = np.diag(pWidth**2)
	C       = np.diag(pWidth**2)
	mean    = deepcopy(start)
	
	scaleInit = np.prod( pWidth**2 )**(1.0/nParam)
	r = 1.0
	
	# all generations
	chain  = []
	chainF = []
	
	M = []
	
	# get initial population
	print "initial solutions"
	popSol    = getInitPop( nPop, nParam, pFit, start, xLim )
	popFit, X = evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b, shift )
	M.append(X)
	print np.sort(popFit)
	print " "
	
	# add init to all
	chain = [ popSol ]
	fit = [ popFit ]
	for i in range(nPop):
		chainF.append( popSol[i] )
	# end
	
	# Random walk
	for step in range(nGen):
		print "step: " + str(step+1) + "/" + str(nGen)
		
		# get covariance matrix
		cov, mean, C = getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nParam, chainF, popSol, r, scaleInit, toMix, mixProb, mixAmp )
		
		# perform selection
		parSol, parFit = Selection( nPop, popSol, popFit, nKeep )
		
		# perform crossover
		popSol = Crossover( nPop, nParam, parSol, parFit, cov, step, nGen )
		
		# perform mutation
		popSol = Mutate( step, nGen, nPop, nParam, popSol, cov, xLim, pFit, toFlip, nFlip, flipProb )
		
		# calculate fits
		popFit, X = evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b, shift )
		
		M.append(X)
		chain.append( popSol )
		fit.append( popFit )
		for i in range(nPop):
			chainF.append( popSol[i] )
		# end
		
		print np.sort(popFit)
		print " "
	# end
	chain = np.array(chain)
	fit   = np.array(fit)
	M     = np.array(M)
	
	return chain, fit, M

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

def cumMin( chain1, fit1 ):
	
	fit2 = []
	chain2 = []
	
	curMax = fit1[0] - 1
	for i in range( len(fit1) ):
		if( fit1[i] > curMin ):
			fit2.append( fit1[i]   )
			chain2.append( chain1[i,:] )
			curMin = fit1[i]
		else:
			fit2.append( fit2[-1]   )
			chain2.append( chain2[-1] )
		# end
	# end
	fit2 = np.array(fit2)
	chain2 = np.array(chain2)
	
	return chain2, fit2
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






