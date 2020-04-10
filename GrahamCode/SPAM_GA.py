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
	
	scrTM     = 2		# which machine score
	scrMU     = 3		# which machine score
	
	nGen      = 2**5	# number of generations
	nPop      = 2**3	# size of population at each step
	nBin      = 65		# bin resolution
	nParam    = 14		# number of SPAM parameters
	sigScale  = 1.0		# scale param stddev for prop width
	initScale = 10.0	# scale param limits
	bound = np.array([	# bin window
#		[-0.6,0.6],
#		[-0.4,0.8]])
		[-19.0, 11.0],
		[-18.0, 12.0]])
	
	targetInd = 0
	
#	pFit = [ 3, 4, 5 ]
	
#	initSeed = 234329	# random seed for initial params
	
	toPlot = 1
	
	shrink = 0.25
	
	
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	# read zoo file
	data, nModel, nCol = ReadAndCleanupData( pFile )
	print " "
	
	pReal = data[targetInd,0:-1]
	
	# get parameter stats
	mins = np.min( data, axis=0 )[0:-1]
	maxs = np.max( data, axis=0 )[0:-1]
#	stds = np.std( data, axis=0 )[0:-1]
	
	xLim = np.zeros((nParam,2))
	for i in range(nParam):
		xLim[i,0] = pReal[i] - shrink*(pReal[i] - mins[i])
		xLim[i,1] = shrink*(maxs[i] - pReal[i]) + pReal[i]
		"""
		xLim[i,0] = pReal[i] - initScale*stds[i]
		if( i >= 6 and xLim[i,0] < 0 ):
			xLim[i,0] = 10**-3
		# end
		xLim[i,1] = pReal[i] + initScale*stds[i]
		"""
	# end
	
	"""
	# manually set
	# position
	stds[0]  = 0.0001
	stds[1]  = 0.0001
	stds[2]  = 0.7
	# velocity
	stds[3]  = 0.1
	stds[4]  = 0.1
	stds[5]  = 0.7
	# mass
	stds[6]  = 0.1
	stds[7]  = 0.7
	# radii
	stds[8]  = 0.1
	stds[9]  = 0.1
	# angles
	stds[10] = 1.3
	stds[11] = 1.3
	stds[12] = 1.3
	stds[13] = 1.3
	"""
	
	# std is a fraction of width
	stds = np.zeros(nParam)
	for i in range(nParam):
		stds[i] = (xLim[i,1]-xLim[i,0])/30.0
	# end
	
	pWidth = stds*sigScale
	
	print "min           real      max        pWidth"
	for i in range(nParam):
		print xLim[i,0], pReal[i], xLim[i,1], pWidth[i]
	# end
	print " "
	
	# get simulated target
	T, V = solve( pReal, nParam, nBin, bound )
	
	# find target perturbedness
	muScore = MachineScore( nBin, T, V, scrMU )
	
	a = muScore
	b = 0.13
	
	
	
	#####$$$$$$#####
	###   MCMC   ###
	#####$$$$$$#####
	
	toMix = 1
	
	burn = nGen/2**-2
	beta = 0.001
	
	mixProb = np.array( [ 1.0/3.0, 1.0/3.0, 1.0/3.0 ] )
	mixAmp  = [ 1.0/3.0, 3.0 ]
	
	# RUN GA -------------
	chain, scores = GA( nGen, nPop, nParam, pReal, xLim, pWidth, nBin, bound, T, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b )
	
	
	pickle.dump( chain,   open("solutions_13.txt", "wb") )
	pickle.dump( scores,  open("scores_13.txt",    "wb") )
	
	
	
	####################
	###   PLOTTING   ###
	####################
	
	if( toPlot == 1 ):
		fig, axes = plt.subplots(nrows=1, ncols=1)
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
#		axes.plot( np.min(  scores, axis=1), 'r-' )
#		axes.plot( np.max(  scores, axis=1), 'r-' )
		axes.plot( np.sort( scores, axis=1), 'r-' )
		
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
	
	return binC, binC_u
	
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
	
	print "score         tm          mu2           muM             muT"
	
	popFit = []
	for i in range(nPop):
#		print popSol[i]
		M, U = solve( popSol[i,:], nParam, nBin, bound )
#		M = np.ones((nBin,nBin))
#		U = np.ones((nBin,nBin))
		
#		tmScore = MachineScore( nBin, T, M, scrTM )
		tmScore1 = MachineScore( nBin, T, M, 0 )
		tmScore2 = MachineScore( nBin, T, M, 2 )
		muScore = MachineScore( nBin, M, U, scrMU )
		
		muScoreX = np.exp( -(muScore - a)**2/(2*b**2))
		score = (tmScore1*tmScore2*muScoreX)**(1.0/3.0)
		
		print score, tmScore1, tmScore2, muScoreX, muScore, a
		
		popFit.append( score )
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
		
#		popFit2 = deepcopy(popFit)
#		popFit2 = popFit2**2
#		popProb = np.cumsum( popFit2/np.sum(popFit2) )
		
		for i in range(nPop/2):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parents.append( [ popSol[ind1], popSol[ind2] ] )
		# end
	#else:
	#	w = 1
	# end
	parents = np.array(parents)
	
	return parents
# end

def Crossover( nPop, nParam, parents ):
	
	crossType = 1
	
	popSol = np.zeros((nPop,nParam))
	
	if(   crossType == 0 ):
		for i in range(nPop/2):
			r  = np.random.uniform(0,1)
			c1 = parents[i,0]*r     + parents[i,1]*(1-r)
			c2 = parents[i,0]*(1-r) + parents[i,1]*r
			popSol[2*i,:]   = c1
			popSol[2*i+1,:] = c2
		# end
	elif( crossType == 1 ):
		for i in range(nPop/2):
			inds = np.random.randint( 2, size=nParam )
			
			c1 = np.zeros(nParam)
			c2 = np.zeros(nParam)
			for j in range(nParam):
				c1[j] = parents[i,  inds[j],j]
				c2[j] = parents[i,1-inds[j],j]
				"""
				if( inds[j] == 0 ):
					c1[j] = parents[i,0,j]
					c2[j] = parents[i,1,j]
				else:
					c1[j] = parents[i,1,j]
					c2[j] = parents[i,0,j]
				# end
				"""
			# end
			popSol[2*i,:]   = c1
			popSol[2*i+1,:] = c2
		# end
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
		
#		for i in range(nPop):
#			print popSol[i]
#			print popFit[i]
#		# end
		
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
	
	print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in A]))
	
# end






main()






