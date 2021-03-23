'''
     Author:    Matthew Ogden
    Created:    22 Mar 2021
Description:    For creating and applying masks
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

def test():
    print("MC: Hi!  You're in masked_image_compare.py")

# For testing new modules from a Jupyter Notebook
test_func = None

def set_test_func( inLink ):
    global test_func
    test_func = inLink

def createEllipse( 
       in_img,    # image adding ellips to
       center,     # Pixel center of the ellipse
       angle = 0,     # The angle of the primary axis
       thickness = 5,     # The thickness of the ellipse
       axesLength = (10,20),     # The lengths of the primary and secondary axis
       color = (1.0,1.0,1.0),     # The color of the ellipse 
       sfAngle = ( 0, 360 ),     # Starting and ending arc angles, (0,360 is full ellipse)
       ):    #

    # Create both Ellipses
    c_img = cv2.ellipse( in_img, center, axesLength, angle, sfAngle[0], sfAngle[1], color, thickness )

    return c_img

# End adding ellipse


def addBinaryMask( in_img, mask ):
    m_img = np.copy( in_img )
    m_img[mask==1.0] = 1.0
    return m_img


def applyBinaryMask( in_img, mask ):
    m_img = np.copy(in_img)
    m_img[ mask < 1.0 ] = 0
    return m_img


def extractBinaryMask( in_img, mask ):
    e_img = np.extract( mask>0.9, in_img )
    return e_img


def mask_binary_simple_compare( img1, img2, mask, cmpArg ):
    
    # Extract elements based on mask
    eimg1 = extractBinaryMask( img1, mask )
    eimg2 = extractBinaryMask( img2, mask )
    
    # Get score using normal direct image comparison functions on extracted elements
    score = -1
    score = dc.createScore( eimg1, eimg2, cmpArg )
    
    return score

# End simple binary mask compare
