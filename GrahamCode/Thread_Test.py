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
	length = 2**2
	
	q1 = Queue()
	q2 = Queue()
	for i in range(length):
		q1.put(i+1)
	# end
	
	ts = []
	for i in range(nProc):
		ts.append( thr.Thread( target=Test, args=( q1,q2,) ) )
		ts[i].start()
	# end
	
	while not q1.empty():
		for i in range(nProc):
			if( not ts[i].is_alive() ):
				ts[i] = thr.Thread( target=Test, args=(q1,q2,) )
				ts[i].start()
			# end
		# end
	# end
	
	for t in ts:
		t.join()
	# end
	print "joined"
	
	for i in range(length):
		print q2.get()
	# end
	
# end

def Test( q1, q2 ):
	x = q1.get()
	q2.put( x )
	time.sleep(0.2)
# end

main()


