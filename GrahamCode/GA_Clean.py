# Graham West
from copy import deepcopy
import sys 
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
from mpl_toolkits.mplot3d import Axes3D

##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	nGen   = 2**8
	nPop   = 2**5
	nDim   = 3
	choice = 2
	
	toMix = 1
	
	mixProb = np.array( [ 1.0/3.0, 1.0/3.0, 1.0/3.0 ] )
	mixAmp  = [ 1.0/5.0, 5.0 ]
	
	burn      = nGen/2**2
	beta      = 0.001
	
	toPlot = 2
	
#	print nGen, burn, burn/2
	
	
	x_start = np.zeros(nDim)
	# Gaussian 
	if( choice == 0 ):
		thresh  = 0.003
		initSig = 0.08
		p       = [0.00,  1.0,0.3,  0.4,2.5, 0.4,2.5 ]
		func = "Gaussian"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.20 )
			dj_sig.append( 0.00 )
		# end
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.0
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
	# rosenbrock
	elif( choice == 1 ):
		thresh  = 0.02
		initSig = 0.4
		p       = [0.00,  1.0,2.0,  0.4,2.5, 0.4,2.5 ]
		func = "Rosenbrock"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-2.0, 2.0] )
			j_sig.append(  0.20 )
			dj_sig.append( 0.04 )
		# end
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 1.0
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
	# Ackley
	elif( choice == 2 ):
		thresh  = 0.8
		initSig = 6.0
		p       = [0.00,  1.0,2.0,  0.4,2.5, 0.4,2.5 ]
		func = "Ackley"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-10.0, 10.0] )
#			xLim.append( [-1000.0, 1000.0] )
#			j_sig.append(  1.0 )
			j_sig.append(  0.3 )
			dj_sig.append( 0.3 )
		# end
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
#			x_start[j] = 8.0
#			x_start[j] = 0.0
			x_start[j] = np.random.uniform(xLim[j][0],xLim[j][1])
		# end
	# bi-modal
	elif( choice == 3 ):
		thresh  = 0.003
		initSig = 0.08
		p       = [0.00,  0.5,0.15,  8.0,0.333, 0.4,2.5 ]
		func = "Bimodal"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.10 )
			dj_sig.append( 0.00 )
		# end
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.8
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
	# many minima, gaussian
	elif( choice == 4 ):
		thresh  = 0.025
		initSig = 0.04
		p       = [0.00,  1.0,3.0,  0.3,2.5, 0.3,2.5 ]
		func = "ManyMins"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.10 )
			dj_sig.append( 0.02 )
		# end
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.8
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
	# V ABS
	elif( choice == 5 ):
		thresh  = 0.025
		initSig = 0.2
		p       = [0.00,  1.0,3.0,  0.3,2.5, 0.3,2.5 ]
		func = "ABS"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.20 )
			dj_sig.append( 0.00 )
		# end
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.0
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
	# end

	# end
	p       = np.array(p)
	xLim    = np.array(xLim)*1.0
	j_sig   = np.array(j_sig)*1.0
	
	progress = 0
	
	
	######################
	###   MCMC STUFF   ###
	######################
	
	chain, error = GA( x_start, p, xLim, j_sig, nGen, nDim, 0, progress, choice, burn, beta, mixAmp, mixProb, toMix, nPop )
	
	minInd1 = np.argmin( np.min( error, axis=1 ) )
	minInd2 = np.argmin( error[minInd1] )
	print "best:"
	print np.min( np.min( error, axis=1 ) ), chain[minInd1,minInd2,:]
	
	
	
	
	####################
	###   PLOTTING   ###
	####################
	
	if( toPlot == 1 ):
		fig, axes = plt.subplots( nrows=1, ncols=2, figsize=(16,9) )
		
		axes = axes.flat
		
		cmap = 'jet'
		for i in range(nGen+1):
			axes[0].scatter( chain[i,:,0], chain[i,:,1], c=error[i,:], s=10, cmap=cmap )
		# end
		
		axes[1].plot( np.mean( error, axis=1 ), 'r' )
		axes[1].plot( np.min( error, axis=1 ), 'g' )
#		axes[1].semilogy( np.mean( error, axis=1 ), 'r' )
#		axes[1].semilogy( np.min( error, axis=1 ), 'g' )
		
		chainMin, errorMin = cumMin( chain[:,0,:], np.min( error, axis=1 ) )
		
#		axes[1].semilogy( errorMin, 'b', linewidth=2.0 )
		axes[1].plot( errorMin, 'b', linewidth=2.0 )
		
		axes[1].set_ylim( [ 0, np.amax(error) ] )
		
	elif( toPlot == 2 ):
#		fig, axes = plt.subplots(nrows=1, ncols=2)
#		fig, axes = plt.subplots(nrows=1, ncols=2, projection='3d')
		fig = plt.figure()
#		ax = fig.add_subplot( 111, projection='3d' )
		
		ax = Axes3D(fig)
		
		i = 0
		j = 1
		k = 2
		
		step = -1
		
#		ax.plot(    chain[:,i], chain[:,j], 'k', linewidth=0.3, zorder=0 )
#		ax.scatter( chain[:,i], chain[:,j], c=scores, cmap='jet', zorder=1 )
		
		for step in range(nGen):
			II = ax.scatter( chain[step,:,i], chain[step,:,j], chain[step,:,k], c=error[step,:], cmap='jet', zorder=1, s=10 )
		# end
#		II = ax.scatter( chain[step,:,i], chain[step,:,j], chain[step,:,k], c=error[step,:], cmap='jet', zorder=1, s=10 )
		cbar = fig.colorbar( II, ax=ax )
#		ax.scatter( chain[:,i], chain[:,j], c=chain[:,5], cmap='jet', zorder=1 )
#		ax.scatter( chain[0,i], chain[0,j], s=250, marker='+', zorder=2 )
		ax.set_xlabel(0)
		ax.set_ylabel(1)
		ax.set_zlabel(2)
	# end
	
	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.tight_layout(w_pad=-0.5, h_pad=-0.5)
	
	plt.show()
	
# end

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


def f( x, a, choice ):
	
	nPar = len(x)
	
#	np.random.seed(int(100*np.abs(x[0])+1000000*np.abs(x[1])))
	
	"""
	q = 0
	for i in range(nPar):
		q += x[i]**2
	# end
	"""
	
	# Gaussian
	if( choice == 0 ):
		q = 0
		for i in range(nPar):
			q += x[i]**2
		# end
		y = a[1]*(1.0-np.exp(-0.5*(q/a[2]**2)**(2.0/2.0)))
	# Rosenbrock
	elif( choice == 1 ):
		y = 0
		for i in range(nPar-1):
			y += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
		# end
	# Ackley
	elif( choice == 2 ):
		A = 20.0
		B = 4.0
		q1 = 0
		q2 = 0
		for i in range(nPar):
			q1 += x[i]**2
			q2 += np.cos(2*np.pi*x[i])
		# end
		y = A*( 1.0 - np.exp(-0.2*(q1/(1.0*nPar))**0.5) ) + B*( np.e - np.exp(q2/(1.0*nPar)) ) + 10**-15
	# bi-modal
	if( choice == 3 ):
		q1 = 0
		q2 = 0
		for i in range(nPar):
			q1 += (x[i] - a[4])**2
			q2 += (x[i] + a[4])**2
		# end
		y = 2*(a[1]*( (1.0-np.exp(-0.5*(q1/a[2]**2)**(a[3]/2.0))) + (1.0-np.exp(-0.5*(q2/a[2]**2)**(a[3]/2.0))) ) - 0.5)
	# many minima, gaussian
	elif( choice == 4 ):
		q = 0
		y = 0
		for i in range(nPar):
			q += x[i]**2
			y += a[3]*(1.0-np.cos(a[4]*2*np.pi*x[i]))
		# end
		y += a[1]*(1.0-np.exp(-0.5*q/a[3]**2))
	# V ABS
	elif( choice == 5 ):
		q = 0
		y = 0
		for i in range(nPar):
			q += x[i]**2
		# end
		y = np.sqrt(q)
	# end
	
	return y
	
# end

def getInitPop( nPop, nDim, xLim ):
	
	# get initial population
	popSol = []
	for i in range(nPop):
		x = np.zeros(nDim)
		for j in range(nDim):
			x[j] = np.random.uniform( xLim[j,0], xLim[j,1] )
		# end
		popSol.append( x )
	# end
	popSol = np.array(popSol)
	
	return popSol
# end

def evalPop( nPop, popSol, choice, p ):
	
	popErr = []
	for i in range(nPop):
		popErr.append( f( popSol[i], p, choice ) )
	# end
	popErr = np.array(popErr)
	
	return popErr
# end

def Selection( nPop, popSol, popErr ):
	
	selectType = 0
	
	parents = []
	
	if( selectType == 0 ):
		sensMult = 0.1
		
		# get selection probabilities
		fitSens = np.mean(popErr)
		popFit  = np.exp( -np.array(popErr)/(fitSens*sensMult) )
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

def Crossover( nPop, nDim, parents ):
	
	crossType = 0
	
	popSol = np.zeros((nPop,nDim))
	
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

def Mutate( nPop, nDim, popSol, cov ):
	
	popSol2 = np.zeros((nPop,nDim))
	for i in range(nPop):
		popSol2[i,:] = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]
	# end
	
	return popSol2
# end

def getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nDim, chainAll, popSol, r, scaleInit, toMix, mixProb, mixAmp ):
	
	chainF = np.array(chainAll)
	
	# get AM cov matrix
	if( step < burn ):
		C = deepcopy(covInit)
	elif( step == burn ):
		mean = np.mean( np.array(chainAll)[nPop*burn/2:,:], axis=0 )
		C    = np.cov( np.transpose( np.array(chainAll)[nPop*burn/2:,:] ) )
#		mean = np.mean( np.array(chainAll)[:,:], axis=0 )
#		C    = np.cov( np.transpose( np.array(chainAll)[:,:] ) )
		
#		r    = np.prod( jump_sigma**2 )**(0.5/nDim)
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
			for i in range(nDim):
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
			w    = scaleInit*w/np.abs(np.prod(w))**(1.0/nDim)
			
			# mix
			for i in range(nDim):
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

def GA( start, p, xLim, jump_sigma, nGen, nDim, tid, progress, choice, burn, beta, mixAmp, mixProb, toMix, nPop ):
	
	covInit = np.diag(jump_sigma**2)
	cov     = np.diag(jump_sigma**2)
	C       = np.diag(jump_sigma**2)
	mean    = deepcopy(start)
	
	scaleInit = np.prod( jump_sigma**2 )**(1.0/nDim)
	r = 1.0
	
	# all generations
	chain  = []
	chainF = []
	
	# get initial population
	popSol = getInitPop( nPop, nDim, xLim )
	popErr = evalPop( nPop, popSol, choice, p )
	
	# add init to all
	chain = [ popSol ]
	error = [ popErr ]
	for i in range(nPop):
		chainF.append( popSol[i] )
	# end
	
#	print "min error gen " + str(0) + ": "
#	print min(popErr)
	
	# Random walk
	for step in range(nGen):
		
		# get covariance matrix
		cov, mean, C = getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nDim, chainF, popSol, r, scaleInit, toMix, mixProb, mixAmp )
		
		# perform selection
		parents = Selection( nPop, popSol, popErr )
		
		# perform crossover
		popSol = Crossover( nPop, nDim, parents )
		
		# perform mutation
		popSol = Mutate( nPop, nDim, popSol, cov )
		
		# calculate errors
		popErr = evalPop( nPop, popSol, choice, p )
		
		chain.append( popSol )
		error.append( popErr )
		for i in range(nPop):
			chainF.append( popSol[i] )
		# end
		
#		print "min error gen " + str(step+1) + ": "
#		print min(popErr)
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
		








main()


