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
import multiprocessing.dummy as dum
import os
import queue as q



##############
#    MAIN    #
##############

b = []

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	nProc  = 2**2
	length = 100
	
	M = np.random.random(length)
	
	p = mp.Process( target=Test, args=( M,) )
	p.start()
	p.join()
	
	print b
	
# end

def Test( a ):
	global b
	b.append(np.sum(a))
	print b
# end

main()


