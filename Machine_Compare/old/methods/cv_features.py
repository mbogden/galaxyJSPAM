'''
       File:    cv_features.py  (Will likely change)
    Version:    v_0     #Still under inital creation
     Author:    Matthew Ogden
    Created:    22 July 2019
    Altered:    22 July 2019
Description:    This program is intended to take two images, extract features
                and compare the likeness of the features.
'''

import numpy as np
import cv2

# Incomplete.  Needs better parameter file to create model images
def harris_corner_compare( img1, img2 ):

    methodName = 'harris_corner_compare_v0_%dx%d' % ( img1.shape[0], img1.shape[1] )

    img1Corners = cv2.cornerHarris( img1, 32, 3, 0.04 )
    img2Corners = cv2.cornerHarris( img2, 32, 3, 0.04 )

    diffImg = np.absolute( img1Corners - img2Corners ) 

    # Scale values to between 0 and 1

    dMin = np.amin( diffImg )
    dMax = np.amax( diffImg )

    diffImg += dMin

    diffImg *= ( 1.0/( dMax+ dMin) )

    sumDiff = np.sum( diffImg )
    avgDiff = sumDiff/diffImg.size
    score = 1 - avgDiff

    return score, methodName, diffImg
# End pixel difference


def hog_compare( img1, img2 ):
    pass

def keypoint_matching( img1, img2 ):

    methodName = 'keypoint_matching_v0'
    diffImg = img1
    score = 0



    return score, methodName, diffImg

# End hog compare
