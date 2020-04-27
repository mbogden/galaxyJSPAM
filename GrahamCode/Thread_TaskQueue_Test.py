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
import Queue
from threading import Thread


##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	nProc  = 2**1
	length = 2**3
	nParam = 2
	
	p = 1
	choice = 2
	
	M = np.random.random((length,nParam))
	
	qy = Queue.Queue()
	
	q = TaskQueue(num_workers=nProc)
	
	for i in range(length):
		q.add_task( f, [M[i,:],i], p, choice, qy )
	# end
	
	q.join()       # block until all tasks are done
	
	out = []
	for i in range(length):
		out.append( qy.get() )
	# end
	out = np.array(out)
	
	popErr = out[out[:,1].argsort(),0]
	print popErr
	
	
	
	
	
	print "All done!"	
	
# end

def f( XXX, a, choice, qy ):
	
	x     = XXX[0]
	index = XXX[1]
	
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
	
#	time.sleep( np.random.random(1)[0] )
#	print y
	
	qy.put( [y,index] )
	
	return y
	
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

main()


