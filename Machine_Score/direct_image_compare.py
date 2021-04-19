'''
     Author:    Matthew Ogden
    Created:    31 Oct 2019
Description:    Created so comparison methods are simply maintained
'''

from inspect import getmembers, isfunction
from os import path
from sys import modules as sysModules, path as sysPath
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# For importing my useful support module
supportPath = path.abspath( path.join( __file__, "../../Support_Code/" ) )
sysPath.append( supportPath )

import general_module as gm

# Populate global list of score functions
scoreFunctions = None

    
def test():
    print("DC: Hi!  You're in direct_image_compare.py")


# For testing new modules from a Jupyter Notebook
test_compare = None

def set_test_compare( inLink ):
    global test_compare
    test_compare = inLink
    updateScoreFunctions()
    print("New Score Function: ",test_compare)

    
def score_test_compare( img1, img2, cmpArg ):
    
    # Check if valid link
    if test_compare == None:
        return -1
    
    else:
        return test_compare( img1, img2, cmpArg )
    

def createScore( img1, img2, cmpArg, printBase=True ):
    
    if img1.dtype != np.float32 or img2.dtype != np.float32:
        if printBase: print("WARNING: DC: Images are not type np.float32.")
        return None
    
    # Check if altering brightness values
    bMatch = cmpArg.get('brightness_match',None)
    
    if bMatch == None:
        pass
    
    if bMatch == 'match_average':
        img2 = matchAvgBrightness( img2, img1 )
        
    elif bMatch == 'match_total':
        img2 = matchTotalBrightness( img2, img1 )

    funcPtr = getScoreFunc( cmpArg['direct_compare_function'] )

    if funcPtr != None:
        return funcPtr( img1, img2, cmpArg )
    
    else:
        return None


# Have images match total brightness
def matchTotalBrightness( img, to_img ):
    sumB = np.sum( img )
    to_sumB = np.sum( to_img )    
    r_img = (to_sumB/sumB) * img
    
    return r_img    
    
def getScoreFunc( funcName, printAll = False ):

    funcPtr = None

    for name, ptr in scoreFunctions:

        if 'score_' + funcName == name:
            funcPtr = ptr
            
        elif name == funcName:
            funcPtr = ptr
    
    return funcPtr
# End get score


def score_absolute_difference( img1, img2, cmpArg ):

    score = None

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size
    score = 1 - score

    # if simply return score
    return score

# End absDiff


def score_absolute_difference_squared( img1, img2, cmpArg ):

    score = None

    dImg = np.power( np.abs( img1 - img2 ), 2 )
    score = np.sum( dImg ) / dImg.size
    score = 1 - score

    # if simply return score
    return score

# End absDiff

def score_ssim( img1, img2, cmpArg ):
    score = -1
    score = ssim( img1, img2 )
    return score

# All basic scoring methods

def binImg( imgIn, threshold ):

    cpImg = np.copy( imgIn )
    cpImg[ cpImg >= threshold] = 1.0
    cpImg[ cpImg < threshold] = 0

    return cpImg


def score_overlap_fraction( img1, img2, cmpArg ):

    score = None    
        
    h1 = cmpArg.get( 'h1', .25 )    
    h2 = cmpArg.get( 'h2', .25 )
    
    bImg1 = binImg( img1, h1 )
    bImg2 = binImg( img2, h2 )

    oImg = bImg1 + bImg2    # Overlap of both images
    oImg = binImg( bImg1 + bImg2, 1.9 )
    
    bImg1, bImg2, oImg = img_overlap_fraction( img1, img2, cmpArg )

    s1 = np.sum( bImg1 )
    s2 = np.sum( bImg2 )
    s3 = np.sum( oImg )

    score = ( s3 / ( s1 + s2 - 1.0*s3 ) )
    
    return score


def score_correlation( img1, img2, cmpArg  ):

    score = None

    if len( img1.shape ) > 1:
        ar1 = img1.flatten()
    else:
        ar1 = img1

    if len( img2.shape ) > 1:
        ar2 = img2.flatten()
    else:
        ar2 = img2

    score = np.corrcoef( ar1, ar2 )[0,1]

    return score


def score_binary_correlation( img1, img2, cmpArg  ):

    score = None
    h1=.25
    h2=.25
    bin1=True
    bin2=True
    
    if cmpArg.get('h1',None) != None:
        h1 = cmpArg['h1']
    
    if cmpArg.get('h2',None) != None:
        h1 = cmpArg['h2']
    
    tmpImg1 = binImg( img1, h1 )
    tmpImg2 = binImg( img2, h2 )

    score = score_correlation( tmpImg1, tmpImg2, cmpArg )

    return score
# end createBinaryCorrelation()


# Have images match average brightness
def matchAvgBrightness( img, to_img ):
    
    avg = np.average( img )
    to_avg = np.average( to_img )    
    r_img = (to_avg/avg) * img
    
    return r_img    

# Gather direct image score functions into a single list
def updateScoreFunctions():
    global scoreFunctions
    scoreFunctions = [ ( name, obj ) for name,obj in getmembers(sysModules[__name__]) \
                    if ( isfunction(obj) and name.split('_')[0] == 'score' ) ]
updateScoreFunctions()

def printScoreFunctions():
    for name, lnk in scoreFunctions:
        print(name, lnk)

if __name__=='__main__':

    test()

    import inspect, sys
    scoreFunctions = [ ( name, obj ) for name,obj in inspect.getmembers(sys.modules[__name__]) \
                        if ( inspect.isfunction(obj) and name.split('_')[0] == 'score' ) ]
    
    #print( scoreFunctions )

    print("Score Functions Found")
    for n,f in scoreFunctions:
        print('\t-',n)
