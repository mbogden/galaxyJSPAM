'''
     Author:    Matthew Ogden
    Created:    05 Apr 2021
Description:    For extracting features from images and comparing
'''

from os import path
from sys import modules as sysModules, path as sysPath
import numpy as np
import cv2

# For importing my useful support module
supportPath = path.abspath( path.join( __file__, "../../Support_Code/" ) )
sysPath.append( supportPath )

import general_module as gm
import direct_image_compare as dc
import masked_image_compare as mc

def test():
    print("MC: Hi!  You're in masked_image_compare.py")

# For testing new modules from a Jupyter Notebook
test_func = None

def set_test_func( inLink ):
    global test_func
    test_func = inLink


def create_feature_score( img1, img2, cmpArg ):
    
    feat_function = cmpArg.get('feature_function', None)
    if feat_function == None:  return None
    
    compare_function = cmpArg.get('direct_compare_function', None)
    if compare_function == None:  return None
    
    if feat_function == 'hog':
        score = score_hog( img1, img2, cmpArg )
        return score


def score_hog_dot( img1, img2, cmpArg ):
    
    # Get HOG features, (Magnitude of change and Angle of change)
    mag1, ang1 = extract_hog( img1, cmpArg )
    mag2, ang2 = extract_hog( img2, cmpArg ) 
    
    v1 = np.zeros(( mag1.shape[0], mag1.shape[1], 2 ) )
    v1[:,:,0]


    if cmpArg.get('non_zero_pixels',False):

        eMag = mc.non_zero_mask( mag1, mag2 )
        eAng = mc.non_zero_mask( ang1, ang2 )
        eIndex = mc.non_zero_mask( eMag, eAng )

        mag1 = mc.extractBinaryMask( mag1, eIndex )
        mag2 = mc.extractBinaryMask( mag2, eIndex )

        ang1 = mc.extractBinaryMask( ang1, eIndex )
        ang2 = mc.extractBinaryMask( ang2, eIndex )
    
    cmp_func = cmpArg.get('direct_compare_function','absolute_difference')
    
    if cmp_func == 'absolute_difference':        

        dMag = np.abs( mag1 - mag2 )
        s1 = np.sum( dMag ) / dMag.size
        s1 = 1 - s1
        
        s2 = np.sum( dAng ) / dAng.size
        s2 = 1 - s2
    
    elif cmp_func == 'absolute_difference_squared':

        dMag = np.abs( mag1 - mag2 )
        dMag = np.power( dMag, 2 )
        s1 = np.sum( dMag ) / dMag.size
        s1 = 1 - s1
        
        dAng = np.power( dAng, 2 )
        s2 = np.sum( dAng ) / dAng.size
        s2 = 1 - s2
        
    else:
        print("FC: Compare function '%s' not available for hog"%cmp_func)
        return None

    w1 = cmpArg.get('weight_mag',0.5)
    w2 = cmpArg.get('weight_ang',0.5)
    score = s1*w1 + s2*w2
    
    return score

        


def score_hog( img1, img2, cmpArg ):
    
    # Get HOG features, (Magnitude of change and Angle of change)
    mag1, ang1 = extract_hog( img1, cmpArg )
    mag2, ang2 = extract_hog( img2, cmpArg ) 
    
    # Normalize magnitude to between 0 and 1.
    maxMag = np.amax( [ np.amax(mag1), np.amax(mag2) ] )
    mag1 /= maxMag
    mag2 /= maxMag
        
    # Find minimum angle difference since it's periodic
    diffAr = np.zeros( (ang1.shape[0], ang1.shape[1], 3) )
    diffAr[:,:,0] = np.abs( ang1 - ang2 )
    diffAr[:,:,1] = np.abs( ang1 - ang2 + np.pi )
    diffAr[:,:,2] = np.abs( ang1 - ang2 - np.pi )
    
    dAng = np.amin(diffAr,axis=2)
    
    # Normalize between 0 and 1, max angle difference is pi

    if cmpArg.get('non_zero_pixels',False):

        eMag = mc.non_zero_mask( mag1, mag2 )
        eAng = mc.non_zero_mask( ang1, ang2 )
        eIndex = mc.non_zero_mask( eMag, eAng )

        mag1 = mc.extractBinaryMask( mag1, eIndex )
        mag2 = mc.extractBinaryMask( mag2, eIndex )

        ang1 = mc.extractBinaryMask( ang1, eIndex )
        ang2 = mc.extractBinaryMask( ang2, eIndex )
    
    cmp_func = cmpArg.get('direct_compare_function','absolute_difference')
    
    if cmp_func == 'absolute_difference':        

        dMag = np.abs( mag1 - mag2 )
        s1 = np.sum( dMag ) / dMag.size
        s1 = 1 - s1
        
        s2 = np.sum( dAng ) / dAng.size
        s2 = 1 - s2
    
    elif cmp_func == 'absolute_difference_squared':

        dMag = np.abs( mag1 - mag2 )
        dMag = np.power( dMag, 2 )
        s1 = np.sum( dMag ) / dMag.size
        s1 = 1 - s1
        
        dAng = np.power( dAng, 2 )
        s2 = np.sum( dAng ) / dAng.size
        s2 = 1 - s2
        
    else:
        print("FC: Compare function '%s' not available for hog"%cmp_func)
        return None

    w1 = cmpArg.get('weight_mag',0.5)
    w2 = cmpArg.get('weight_ang',0.5)
    score = s1*w1 + s2*w2
    
    return score


        

# Image HOG Comparison.  Code copied and modified from the following link:
# https://learnopencv.com/histogram-of-oriented-gradients/
def extract_hog( img, cmpArg ):
    
    ks = int( cmpArg.get('kernal_size',1) )

    gx = cv2.Sobel( img, cv2.CV_32F, 1, 0, ksize = ks )
    gy = cv2.Sobel( img, cv2.CV_32F, 0, 1, ksize = ks )

    mag, angle = cv2.cartToPolar( gx, gy, angleInDegrees=False )
    
    return mag, angle
