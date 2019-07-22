'''
       File:    pixel_difference.py
    Version:    v_0     #Still under inital creation
     Author:    Matthew Ogden
    Created:    22 July 2019
    Altered:    22 July 2019
Description:    This program holds functions for comparing two images 
                by directly comparing pixels of the images in some way.
'''

import numpy as np

def direct_difference( img1, img2 ):

    methodName = 'pixel_difference_v0_%dx%d' % ( img1.shape[0], img1.shape[1] )

    diffImg = np.absolute( img1 - img2 ) 

    sumDiff = np.sum( diffImg )

    avgDiff = sumDiff/diffImg.size

    score = 1 - avgDiff

    return score, methodName, diffImg
