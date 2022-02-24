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

import cv2
import multiprocessing as mp
import threading as thr
import os
import time
from Queue import Queue



##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	nProc  = 2**1
	length = 2**3
	nParam = 3
	
	p = 1
	choice = 2
	
	
	q1 = Queue()
	q2 = Queue()
	for i in range(length):
		q1.put( [i, np.random.random(nParam)] )
	# end
	
	ts = []
	for i in range(nProc):
		ts.append( thr.Thread( target=f, args=(q1, q2, p, choice,) ) )
		ts[i].start()
	# end
	
	while not q1.empty():
#		print q1.qsize()
		for i in range(nProc):
			if( not ts[i].is_alive() ):
				ts[i].join()
				ts[i] = thr.Thread( target=f, args=(q1, q2, p, choice,) )
				ts[i].start()
			# end
		# end
	# end
	
	out = []
	for i in range(length):
		out.append( q2.get() )
	# end
	out = np.array(out)
	popErr = out[out[:,0].argsort(),1]
	print popErr
	
	print "about to join"
	for t in ts:
		print t.name + " joining " + str(t.is_alive())
		t.join()
		print t.name + " joined "  + str(t.is_alive())
	# end
	print thr.enumerate()
	
# end

def f( Q1, Q2, a, choice ):
	
	XXX   = Q1.get()
	index = XXX[0]
	x     = XXX[1]
	
	nPar = len(x)
	
	# Gaussian
	if( choice == 0 ):
		q = 0
		for i in range(nPar):
			q += x[i]**2
		# end
		y = a[1]*(1.0-np.exp(-0.5*(q/a[2]**2)**(2.0/2.0)))
	elif( choice == 2 ):
		A = 20.0
		B = 4.0
		s1 = 0
		s2 = 0
		for i in range(nPar):
			s1 += x[i]**2
			s2 += np.cos(2*np.pi*x[i])
		# end
		y = A*( 1.0 - np.exp(-0.2*(s1/(1.0*nPar))**0.5) ) + B*( np.e - np.exp(s2/(1.0*nPar)) ) + 10**-15
	# end
	
	time.sleep( np.random.random(1)[0] )
	
	Q2.put( [index, y] )
#	return y
	
# end

def Test( q1, q2 ):
	q2.put( q1.get() )
# end

main()


