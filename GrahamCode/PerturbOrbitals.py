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

import cv2
import pickle


##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	zooInd = "587722984435351614"
#	zooInd = "587729227151704160"
	
	nParam    = 14		# number of SPAM parameters
	bound = np.array([	# bin window
#		[-0.6,0.6],
#		[-0.4,0.8]])
		[-19.0, 11.0],
		[-18.0, 12.0]])
	
	pTest = 5
	
	direct   = ""
	fileBase = "PerturbOrbitals_" + str(zooInd)
	fileID   = "00"
	filePPG  = "2000ppg"
	
	tm = 4
	mu = 4
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	fileP = fileBase + "_P_" + filePPG + "_" + fileID + ".pk"
	fileM = fileBase + "_M_" + filePPG + "_" + fileID + ".pk"
	fileU = fileBase + "_U_" + filePPG + "_" + fileID + ".pk"
	
	pVals = pickle.load( open(direct + fileP, "rb") )
	M     = pickle.load( open(direct + fileM, "rb") )
	U     = pickle.load( open(direct + fileU, "rb") )
	
	shp   = M.shape
	nGen  = shp[0]
	nBin  = shp[1]
	print pVals.shape
	
	mASt = 0
	
	T = M[mASt,:,:]
	V = U[mASt,:,:]
	
	
	
	###################
	###   FITNESS   ###
	###################
	
	scr = 2
	
	nScr = 5
	
	nTM = nScr
	nMU = nScr
	
	# find target SPAM paramt
#	x, y, a, DD = MachineScore(      nBin, T, T, U, 1, 1, scr )
	a = 1
	b = 0.17
	
	score    = np.zeros((nGen,nTM,nMU))
	tmScore  = np.zeros((nGen,nScr))
	muScore  = np.zeros((nGen,nScr))
	muScore2 = np.zeros((nGen,nScr))
	
	# get error profiles
	for i in range(nGen):
		for j in range(nScr):
			tmScore[i,j] = MachineScore( nBin,    T, M[i], a, b, j )
			muScore[i,j] = MachineScore( nBin, M[i], U[i], a, b, j )
		# end
	# end
	
	for i in range(nScr):
		muScore2[:,i] = np.exp( -(muScore[:,i] - muScore[mASt,i])**2/(2*b**2) )
		for j in range(nScr):
			score[:,j,i] = tmScore[:,j]*muScore2[:,i]
#			score[:,j,i] = tmScore[:,j]*tmScore[:,i]
		# end
	# end
	
	
	####################
	###   PLOTTING   ###
	####################
	
	toPlot = 1
	
	plotUnp = 1
	
	ms = 1
#	ls = 1
	ls = 1.5
	
	labels = [ 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mr', 'mt', 'rp', 'rs', 'pp', 'ps', 'tp', 'ts', 'tmin', 'dmin', 'vmin', 'beta', 'inc', 'argPer', 'longAN']
	
	if(   toPlot == 1 ):
		fig, axes = plt.subplots(nrows=4, ncols=5)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		for i in range(2,nParam+7):
			axes[i-2].plot(  [ pVals[0,i], pVals[0,i] ], [0, 1], 'r' )
			axes[i-2].plot( pVals[:,i], score[:,tm,mu], 'b.' )
			axes[i-2].set_xlabel( labels[i] )
			axes[i-2].set_ylim( [0, 1] )
		# end
		
		
		
	elif( toPlot == 2 ):
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
#		fig.set_size_inches(15,10)
		fig.set_size_inches(12,8)
		
		
		for i in range(nScr):
			axes[i].plot( tmScore[:,i] )
			axes[i+8].plot( muScore2[:,i] )
		# end
		
	elif( toPlot == 3 ):
		fig, axes = plt.subplots(nrows=2, ncols=2)
#		axes = axes.flatten()
#		fig.set_size_inches(15,10)
		fig.set_size_inches(12,8)
		
		ms = 3
		
		"""
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		U[U>h] = 1
		"""
		
#		axes[0,0].imshow( T, interpolation='none', cmap='gray' )
		axes[0,0].plot( RVt[0:nPts/2,0], RVt[0:nPts/2,1], 'b.' )
		axes[0,0].plot( RVt[nPts/2:-1,0], RVt[nPts/2:-1,1], 'r.' )
		
		jj = 2
		kk = 3
		
		axes[0,1].plot( pVals[:,pTest], score[:,jj,kk], 'r-' )
		axes[0,1].set_ylim( 0, 1 )
		
		for i in range(nGen):
#			axes[0,1].clear()
			axes[1,0].clear()
			axes[1,1].clear()
			
#			axes[1,0].imshow( M[i,:,:], interpolation='none', cmap='gray' )
			axes[1,0].plot( RVm[i,0:nPts/2,0], RVm[i,0:nPts/2,1], 'b.' )
			axes[1,0].plot( RVm[i,nPts/2:-1,0], RVm[i,nPts/2:-1,1], 'r.' )
			
#			axes[1,1].imshow( U[i,:,:], interpolation='none', cmap='gray' )
			axes[1,1].plot( RVu[i,0:nPts/2,0], RVu[i,0:nPts/2,1], 'b.' )
			axes[1,1].plot( RVu[i,nPts/2:-1,0], RVu[i,nPts/2:-1,1], 'r.' )
			
			axes[0,1].plot( pVals[i,pTest], score[i,jj,kk], 'b.' )
			
			plt.pause(0.5)
		# end

		
	# end
	
	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.show()
	
	
##############
#  END MAIN  #
##############


def MachineScore( nBin, binCt, binCm, a, b, scr ):
	
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
		
		tmScore = 0.0
		for i in range(nBin):
			for j in range(nBin):
				tmScore += ( X[i,j] - Y[i,j] )**2
			# end
		# end
		
		s = 4
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
		tmSCore = tmScore**2
#		tmScore = ( ((tmScore+1)/2.0) )**2
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

def SetProposalWidthsAndParameterRanges( min_, max_, std, param_real, nParam, p_toFit, sigFac, threshInd ):
	
	# sigma for MCMC steps
	jump_sigma = np.zeros([nParam])
	pLim       = np.zeros([nParam,2])
	
	# set proposal widths
	print "proposal widths"
	for i in range(nParam):
		if( i in p_toFit ):
			jump_sigma[i] = sigFac*std[i-2,threshInd]
			print( "{:.5f} {:.5f}".format(jump_sigma[i], std[i-2,threshInd]) )
		else:
			jump_sigma[i] = 0
			if( i >= 2 ):
				print( "{}     {:.5f}".format(jump_sigma[i], std[i-2,threshInd]) )
			else:
				print( "{}     xxx".format(jump_sigma[i]) )
			# end
		# end
	# end
	print " "
	
	# set pLim
	for i in range(2,nParam):
#		pLim[i,0]    = -np.inf
#		pLim[i,1]    =  np.inf
		pLim[i,0]    = min_[i-2,threshInd]
		pLim[i,1]    = max_[i-2,threshInd]
	# end
#	pLim[6,0]  = 0.0
#	pLim[7,0]  = 0.0
	pLim[0,0]  = -np.inf
	pLim[0,1]  =  np.inf
	pLim[1,0]  = -np.inf
	pLim[1,1]  =  np.inf
#	print pLim
	
	return jump_sigma, pLim
	
# end

def GetInitialParameters( param_real, nParam, pFit, median, std, mult, threshInd, initSeed, pFile ):
	
	
	# read target's parameters
	paramFile = open(pFile)
	for i in range(1):
		paramStr = paramFile.readline().split('\t')[1]
#		print paramStr
		param_start = [float(it) for it in paramStr.split(',')]
	# end
	
	# convert to mass ratio/total mass
	r = param_start[6]/param_start[7]
	t = param_start[6]+param_start[7]
	param_start[6] = r
	param_start[7] = t
	
	param_start = np.array(param_start)[0:nParam]
	for i in range(nParam):
		if( i not in pFit ):
			param_start[i] = param_real[i]
		# end
	# end
	
	"""
	# init. initial params
	param_start = np.zeros(nParam)
	
	# generate same initial params...
	randSeed = int(10000000*np.random.uniform(0,1))
	np.random.seed(initSeed)
	
	for i in range(nParam):
		if( i not in p_toFit ):
			param_start[i] = param_real[i]
		else:
			param_start[i] = param_real[i] + np.random.normal(0,mult*std[i-2,threshInd])
		# end
	# end
	
	# ...but allow different paths
	np.random.seed(randSeed)
	"""
	
	
	print "initial - target"
	for i in range(nParam):
		if( i in pFit ):
			print( "{:.5f} {:.5f} ***".format(param_start[i], param_real[i]) )
		else:
			print( "{:.5f} {:.5f}".format(param_start[i], param_real[i]) )
		# end
	# end
	print " "
	
	return param_start
	
# end

def ReadTargetData( nBin, nParam, bound, pFile ):
	
	# read target's parameters
	paramFile = open(pFile)
	paramStr = paramFile.readline().split('\t')[1]
#	print paramStr
	param_real = [float(it) for it in paramStr.split(',')]
	
	# convert to mass ratio/total mass
	r = param_real[6]/param_real[7]
	t = param_real[6]+param_real[7]
	param_real[6] = r
	param_real[7] = t
	
	param_real = np.array(param_real)[0:nParam]
	
	return param_real
	
# end

def ReadParameterRanges( n, m ):
	
	with open('StatsFile.txt', 'r') as f:
		lines = f.readlines()
	
	num = map( float, lines[0].split(', ') )
	
	min_   = []
	max_   = []
	median = []
	mean   = []
	std    = []
	
	for i in range(m):
		min_.append(   map( float, lines[1+5*i].split(', ') ))
		max_.append(   map( float, lines[2+5*i].split(', ') ))
		median.append( map( float, lines[3+5*i].split(', ') ))
		mean.append(   map( float, lines[4+5*i].split(', ') ))
		std.append(    map( float, lines[5+5*i].split(', ') ))
	# end
	
	num    = np.array( num )
	min_   = np.array( min_ )
	max_   = np.array( max_ )
	median = np.array( median )
	mean   = np.array( mean )
	std    = np.array( std )

	return num, min_, max_, median, mean, std
	
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

def SmoothImg( img1, nBin1, w ):
	
	# w = window width
	
	stride = 1
	
	nBin2 = nBin1 - w + 1
	
	img2 = np.zeros((nBin2,nBin2))
	
	for i in range(nBin2):
		for j in range(nBin2):
			img2[i,j] = np.mean( img1[i:i+w,j:j+w] )
		# end
	# end
			
	return img2
	
# end

def Step( x, h ):
	if( x > h ):
		return 1
	else:
		return 0
	# end
# end

def Delta( x ):
	if( x == 0 ):
		return 1
	else:
		return 0
	# end
# end

"""
def MachineScore( nBin, binCt, M, M, binCu, a, b ):
	
	corr = 2
	
	if( corr == 0 ):
		M = max( 1.0*np.amax(binCt), 1.0*np.amax(M) )
		binCt  /= M
		M /= M
		
		tmScore = 1.0
		for i in range(nBin):
			for j in range(nBin):
				
				tmScore += ( binCt[i,j] - M[i,j] )**2
				
			# end
		# end
		tmScore /= 1.0*nBin**2
		tmScore = 1 - tmScore**0.5
		
		
		M  = M.flatten()
		binCu  = binCu.flatten()
		muScore = np.corrcoef( M, binCu )[0,1]
	elif( corr == 1 ):
		h = 20
		
		tmScore = 0
		for i in range(nBin):
			for j in range(nBin):
				
#				tmScore += Delta( Delta( binCt[i,j] ) - Delta( M[i,j] ) )
				tmScore += Delta( Step( binCt[i,j], h ) - Step( M[i,j], h ) )
				
			# end
		# end
		tmScore /= 1.0*nBin**2
		
		
		M  = M.flatten()
		binCu  = binCu.flatten()
		muScore = np.corrcoef( M, binCu )[0,1]
	elif( corr == 2 ):
		M = max( 1.0*np.amax(binCt), 1.0*np.amax(M) )
		h = M/2
		
		x = 0
		y = 0
		z = 0
		for i in range(nBin):
			for j in range(nBin):
				if( binCt[i,j] > h ):
					x += 1
				# end
				if( M[i,j] > h ):
					y += 1
				# end
				if( M[i,j] > h and binCt[i,j] > h ):
					z += 1
				# end
			# end
		# end
		tmScore = (z / (x + y - z*1.0) )
		
		M  = M.flatten()
		binCu  = binCu.flatten()
		muScore = np.corrcoef( M, binCu )[0,1]
	elif( corr == 3 ):
		M = max( 1.0*np.amax(binCt), 1.0*np.amax(M) )
		binCt  /= M
		M /= M
		
		z = 0
		tot = 0.0
		for i in range(nBin):
			for j in range(nBin):
				if( M[i,j] > h and binCt[i,j] > h ):
					z += 1
				# end
				tot += min( binCt[i,j], M[i,j] )
			# end
		# end
		tmScore = tot/(1.0*z)
		
		M  = M.flatten()
		binCu  = binCu.flatten()
		muScore = np.corrcoef( M, binCu )[0,1]
	elif( corr == 4 ):
#		binCt2 = binCt2.flatten()
#		M = M.flatten()
	
		binCt  = binCt.flatten()
		M  = M.flatten()
		M  = M.flatten()
		binCu  = binCu.flatten()
		
#		tmScore = np.corrcoef( binCt2, M )[0,1]
		tmScore = np.corrcoef( binCt, M )[0,1]
		muScore = np.corrcoef( M, binCu )[0,1]
	# end
	
	score   = tmScore * np.exp( -( (muScore - a)**2 )/( 2*b**2 ) )
	
	return score, tmScore, muScore
	
# end
"""

"""
def MachineScore( nBin, binCt, binCm, binCu, a, b ):
	
#	binCt2 = binCt2.flatten()
#	M = M.flatten()
	
	binCt  = binCt.flatten()
	binCm  = binCm.flatten()
	binCu  = binCu.flatten()
	
#	tmScore = np.corrcoef( binCt2, M )[0,1]
	tmScore = np.corrcoef( binCt, binCm )[0,1]
	muScore = np.corrcoef( binCm, binCu )[0,1]
	
	score   = tmScore * np.exp( -( (muScore - a)**2 )/( 2*b**2 ) )
	
	return score, tmScore, muScore
	
# end
"""

"""
def MachineScore( nBin, binCt, binCm, binCu, a, b ):
	
	binCt = binCt.flatten()
	binCm = binCm.flatten()
	binCu = binCu.flatten()
	
d	# brightness correlation
	tmScore1 = np.corrcoef( binCt, binCm )[0,1]
	muScore1 = np.corrcoef( binCm, binCu )[0,1]
	
	# binary correlation
	binCt[ binCt > 0 ] = 1
	binCm[ binCm > 0 ] = 1
	binCu[ binCu > 0 ] = 1
	
	tmScore2 = np.corrcoef( binCt, binCm )[0,1]
	muScore2 = np.corrcoef( binCm, binCu )[0,1]
	
	tmScore = 0.5*(tmScore1 + tmScore2)
	muScore = 0.5*(muScore1 + muScore2)
	
	score   = tmScore * np.exp( -( (muScore - a)**2 )/( 2*b**2 ) )
	
	return score, tmScore, muScore
	
# end
"""

def log_likelihood( param, nParam, binCt, nBin, bound, a, b ):
	
	binCm, binCu = solve( param, nParam, nBin, bound )
	
	score, tmScore, muScore = MachineScore( nBin, binCt, binCm, binCu, a, b )
#	print score, tmScore, muScore
	
	if( score > 0 ):
#		ll = -( np.log(2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
		ll = 5*np.log( score )
	else:
		ll = -np.inf
	# end
	
	return ll, score
	
# end

def log_prior( param, pLim ):
	
	inRange = 1
	# num-1: don't worry about sigma range
	for i in range(len(param)):
		if( not ( pLim[i,0] <= param[i] <= pLim[i,1] ) ):
			inRange = inRange*0
#		print i, pLim[i,0] <= param[i] <= pLim[i,1], pLim[i,0], param[i], pLim[i,1]
#	print inRange
	if( inRange ):
		return 0
	else:
		return -np.inf
	# end
	
# end

def log_posterior( param, nParam, binCt, nBin, bound, a, b, pLim ):
	
	log_pri = log_prior(param, pLim)
#	print log_pri
	if np.isfinite(log_pri):
		log_like, score = log_likelihood( param, nParam, binCt, nBin, bound, a, b )
		return log_pri + log_like, score
	else:
		return -np.inf, -1234
	# end
	
# end

def MCMC( start, nParam, binCt, nBin, pLim, pWidth, nStep, pFit, burn, beta, r, P, bound, a, b, toMix, mixProb, mixAmp ):
	
	n_accept = 0.0
	
	covInit = np.diag(pWidth**2)
	cov     = np.diag(pWidth**2)
	mean    = deepcopy(start)
	
	y_i = np.random.uniform( low=0, high=1, size=nStep )
	
	chain = [ start ]
	cur   = chain[-1]
	cand  = chain[-1]
	
	cur_lp, cur_score = log_posterior( start, nParam, binCt, nBin, bound, a, b, pLim )
	
	lps    = [ cur_lp ]
	scores = [ cur_score ]
	print "initial score: " + str(scores[-1])
	print " "
	
	# Random walk
	for step in range(nStep):
		print str(step+1) + "/" + str(nStep)
		cur = chain[-1]
		
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
#				w    = w/np.prod(w)**(1.0/nParam)
				
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
				W  = np.diag(w)
#				cov = np.dot( np.dot(v,W), LA.inv(v) )
				cov = np.dot( np.dot(v,W), np.transpose(v) )
				cov = cov.real
				cov2 = 0.5*( cov + np.transpose(cov) )
			# end
		# end
		
		cand = np.random.multivariate_normal(mean=cur,cov=cov,size=1)[0]
		
		cand_lp, cand_score = log_posterior( cand, nParam, binCt, nBin, bound, a, b, pLim )
		accProb = np.exp((cand_lp - cur_lp))
		accProb = min( 1, accProb )
		
		if( y_i[step] <= accProb ):
			n_accept += 1
			cur_lp = cand_lp
			cur_score = cand_score
			cur = deepcopy(cand)
			chain.append(cur)
			lps.append( cur_lp )
			scores.append( cur_score )
			print str(scores[-1]) + "    " + str(np.amax(scores))
			print "   accepted: " + str(accProb)
		else:
			chain.append(cur)
			lps.append( cur_lp )
			scores.append( cur_score )
			print str(scores[-1]) + "    " + str(np.amax(scores))
			print "rejected   : " + str(accProb)
		# end
#		print cand_lp, cur_lp
		
		if( step == burn ):
			mean  = np.mean( np.array(chain)[burn:,:], axis=0 )
			C     = np.cov( np.transpose( np.array(chain)[burn:,:] ) )
			
			cov   = r**2*C + beta*covInit
			r     = r*np.exp( ( (1.0*n_accept)/step - P ) )
		elif( step > burn ):
			gamma = 1.0/(step+1)
			dx    = cur  - mean
			mean  = mean + gamma*dx
			C     = C    + gamma*(np.outer(dx,dx) - C)
			
			cov   = r**2*C + beta*covInit
			r     = r*np.exp( gamma*(accProb - P) )
		# end
		print "r: " + str(r)
		
#		for i in pFit:
#			print pLim[i,0], cand[i], pLim[i,1]
		# end
		
		print " "
	# end
	
	# Acceptance rate
	accRate = (1.0*n_accept/nStep)
	print "accRate: " + str(accRate)
	
	chain = np.array(chain)
	lps   = np.array(lps)
	
	return chain, lps, scores

# end

def metropolis( start, nParam, binCt, nBin, pLim, jump_sigma, nStep, toMix, mixAmp, bound, alpha ):
	
	cov = np.diag(jump_sigma**2)
	zero = np.zeros(nParam+1)
	mixProb = np.array( [1.0/3.0, 1.0/3.0, 1.0/3.0] )
	n_accept = 0.
	
	# Draw n random samples from Gaussian proposal distribution to use for 
	# determining step size
	jumps = np.random.multivariate_normal(mean=zero,
	                                      cov=cov, 
	                                      size=n)
	
	for i in range(nStep):
		for j in range(len(start)):
			if( jump_sigma[j] == 0.0 ):
				jumps[i][j] = 0.0
	
	# Draw n random samples from U(0, 1) for determining acceptance
	y_i = np.random.uniform(low=0, high=1, size=n)
	
	# Create a chain and add start position to it
	chain = np.array([start,])
	cur   = chain[-1]
	cand  = chain[-1]
	cur_lp = log_posterior( start, nParam, data_real, nBin, pLim, bound, alpha )
	print cur_lp
	
	max_lp = cur_lp
	max_ps = start
	
	isAcc = 1
	
	wFile = open('Metropolis_ParamsOut.txt', 'w')
	p = deepcopy(cur)
	a = p[6]
	b = p[7]
	p[7] = b/(a+1)
	p[6] = a*p[7]
	wFile.write(','.join(map(str,p))+' '+str(0)+' '+str(isAcc)+' '+str(cur_lp)+'\n')
	
	# Random walk
	for step in range(n):
		print "step: ", step
		
		"""
		print "cand", cand[:15]
		print "cur",  cur[:15]
		print "jump", jumps[step,:15]
		"""
		
		# Get current position of chain
		cur = chain[-1]
			
		if( toMix == 0 ):
			cand = cur + jumps[step]
		else:
			for i in range(nParam):
				r = np.random.uniform(0,1)
				mod = 1.0
				
				# thinning
				if( r <= mixProb[0] ):
					jumps[step][i] *= mixAmp[0]
					cand[i] = cur[i] + jumps[step][i]
#					print i, curDec[i], mod
				# widening
				elif( r <= mixProb[0] + mixProb[1] ):
					jumps[step][i] *= mixAmp[1]
					cand[i] = cur[i] + jumps[step][i]
				# fixing
				else:
					cand[i] = cur[i] + jumps[step][i]
				# end
			# end
			cand[-1] = cur[-1] + jumps[step][-1]
		# end
		
		cand_lp = log_posterior(cand, nParam, data_real, nBin, pLim, bound, alpha )
		acc_prob = np.exp(cand_lp - cur_lp)
		
		# Accept candidate if y_i <= alpha
		if y_i[step] <= acc_prob:
			n_accept += 1
			cur = cand
			cur_lp = cand_lp
			print cur_lp
			chain = np.append(chain, [cur,], axis=0)
			isAcc = 1
			#chain.append(cand)
			print "      accepted"
		else:
			chain = np.append(chain, [cur,], axis=0)
			isAcc = 0
			print cand_lp
			print "rejected"
		
		p = deepcopy(cand)
		a = p[6]
		b = p[7]
		p[7] = b/(a+1)
		p[6] = a*p[7]
		wFile.write(','.join(map(str,p))+' '+str(step+1)+' '+str(isAcc)+' '+str(cur_lp)+'\n')
		
		if( cur_lp > max_lp ):
			max_lp = cur_lp
			max_ps = cur
	# end
	
	wFile.close()
	
	# Acceptance rate
	acc_rate = (100*n_accept/n)
		
	return [chain, acc_rate, max_ps, max_lp]
	
# end

def ErrorFunction_Old( nBin, Vi, V1 ):
	
	As = 0.0
	At = 0.0
	Ovr = 0.0
	MSE = 0.0
	error = 0.0
	
	isOvr = 0
	
	W = np.ones((nBin,nBin))
	
	for i in range(nBin):
		for j in range(nBin):
		#	print ( Vi[j,i] - V1[j,i] )**2
			
			if( V1[j,i] > 0 or V1[j,i] < 0 ):
				At = At + 1
				if( Vi[j,i] > 0 or Vi[j,i] < 0 ):
					As = As + 1
					Ovr = Ovr + 1
					isOvr = 1
			elif( Vi[j,i] > 0 or Vi[j,i] < 0 ):
				As = As + 1
			# end if
			
			if( isOvr == 1 ):
				MSE = MSE + ( Vi[j,i] - V1[j,i] )**2
			# end if
			
			isOvr = 0
		
		# end for
	# end for
	
	if( Ovr > 1 ):
		MSE = MSE/Ovr
	# end
	
	RMSE = math.sqrt(MSE)
	
	c = 1
	OvrFrac = Ovr/(As+At-Ovr)
#	OvrFrac = Ovr/(As)
#	RMSE = OvrFrac**2*RMSE + (1-OvrFrac**2)*c*RMSE
	
	# final error result
	error = (1 - OvrFrac)*RMSE
	
	return error, OvrFrac, RMSE
	
# end

def BinField_Vel( nBin, RV, bound, alpha ):

	nPts = np.size(RV,0)-1
	
	binC  = np.zeros((nBin,nBin,nBin))
	binVel  = np.zeros((nBin,nBin,nBin))
	
	binC2 = np.zeros((nBin,nBin))
	binVel2 = np.zeros((nBin,nBin))

	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	zmin = np.min( RV[:,2] )
	zmax = np.max( RV[:,2] )
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	dz = (zmax-zmin)/nBin
	
	for i in range(nPts):
		x = float(RV[i,0])
		y = float(RV[i,1])
		z = float(RV[i,2])
		vz = float(RV[i,5])
		
		ii = int( (x - xmin) / (xmax - xmin) * nBin )
		jj = int( (y - ymin) / (ymax - ymin) * nBin )
		kk = int( (z - zmin) / (zmax - zmin) * nBin )
		
		if( ii > 0 and ii < nBin and jj > 0 and jj < nBin and kk > 0 and kk < nBin ):
		        binC[jj,ii,kk] = binC[jj,ii,kk] + 1
		        binVel[jj,ii,kk] = binVel[jj,ii,kk] + vz
		# end
	# end
	
	for i in range(nBin):
		for j in range(nBin):
			for k in range(nBin):
				if( binC[i,j,k] > 1 ):
					binVel[i,j,k] = binVel[i,j,k]/binC[i,j,k]
			# end
		# end
	# end
	
	
	for i in range(nBin):
		for j in range(nBin):
			sumV = 0
			sumW = 0
			
			for k in range(nBin):
				if( binC[i,j,k] > 0 ):
					w = np.exp(-alpha * np.sum(binC[i,j,nBin-k-1:nBin])*dz/(dx*dy*dz)/(nPts) )
					
					sumV += binC[i,j,k]*w*binVel[i,j,k]
					sumW += binC[i,j,k]*w
				# end
			# end
			if( np.sum(binC[i,j]) > 0 ):
				binVel2[i,j] = sumV/sumW
			else:
				binVel2[i,j] = 0
			# end
			binC2[i,j] = np.sum(binC[i,j,:])
		# end
	# end
	
#	return binC, binVel, binC2, binVel2
	return binC2, binVel2
	
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






