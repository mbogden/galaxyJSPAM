'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    11 Oct 2019
Description:    Image creator for use in mdoel to score pipeline. Currently in developement.

'''

from sys import \
        exit, \
        argv

from os import \
        path,\
        listdir, \
        system

import numpy as np
import cv2

def addCircle( img, point, color=(255,255,255), radius = 10, thickness = 2 ):
    cv2.circle( img, point, radius, color, thickness ) 
    return img

def test():
    print( "Inside Misc Image functions" )

# Run main after declaring functions
if __name__ == "__main__":
    argList = argv
    image_creator_pl_v1( argList )


