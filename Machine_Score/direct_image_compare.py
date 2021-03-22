'''
     Author:    Matthew Ogden
    Created:    31 Oct 2019
    Altered:    21 Feb 2020
Description:    Created so comparison methods are simply maintained
'''

from inspect import getmembers, isfunction
from sys import modules as sysModules

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

#from skimage.metrics import structural_similarity as ssim

def test():
    print("DC: Hi!  You're in direct_image_compare.py")
    

# Populate global list of score functions
scoreFunctions = None

    
def createScore( img1, img2, cmpArg ):

    funcPtr = getScoreFunc( cmpArg['function_name'] )

    if funcPtr != None:
        return funcPtr( img1, img2, cmpArg )
    
    else:
        return None
    
    
def getScoreFunc( funcName, printAll = False ):

    funcPtr = None

    for name, ptr in scoreFunctions:

        if 'score_' + funcName == name:
            funcPtr = ptr
            
        elif name == funcName:
            funcPtr = ptr
    
    return funcPtr
# End get score

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
    

# TODO
# For testing new mask functions from Jupyter Notebook
test_mask = None

def set_test_mask( inLink ):
    global test_mask
    test_mask = inLink
    

def mask_test_compare( img1, img2, mask, cmpArg ):
    
    # Check if valid link
    if test_compare == None:
        return 0
    
    else:
        return test_compare( img1, img2, cmpArg )
# TODO    


def score_absolute_difference( img1, img2, cmpArg ):

    score = None

    # adjust so mean brightness matches

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score

    # if simply return score
    return score

# End absDiff

def score_absolute_difference_lower_scale( img1, img2, cmpArg  ):

    score = None
    
    l = 0.9
    
    if cmpArg.get('l',None) != None:
        l = cmpArg['l']

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score

    # arbitrary adjustment to make score more human readable 
    score = (score - l) / (1-l)
    if score < 0.0: score = 0.0

    return score

# end createMScores()

def score_ssim( img1, img2, cmpArg ):
    score = -1
    score = ssim( img1, img2 )
    return score

# All basic scoring methods

def score_overlap_fraction( img1, img2, cmpArg ):

    score = None
    
    h1 = 80
    h2 = 80
    
    if cmpArg.get('h1',None) != None:
        h1 = cmpArg['h1']
    
    if cmpArg.get('h2',None) != None:
        h1 = cmpArg['h2']

    i1 = np.copy( img1 )
    i2 = np.copy( img2 )

    i1[ i1 <  h1 ] = 0
    i1[ i1 >= h1 ] = 1

    i2[ i2 <  h2 ] = 0
    i2[ i2 >= h2 ] = 1

    bImg = i1 + i2
    bImg[ bImg <  2 ] = 0
    bImg[ bImg >= 2 ] = 1

    x = np.sum( i1 )
    y = np.sum( i2 )
    z = np.sum( bImg )

    score = ( z / ( x + y - 1.0*z ) )
    
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

def binImg( imgIn, threshold ):

    cpImg = np.copy( imgIn )
    cpImg[ cpImg >= threshold] = 255
    cpImg[ cpImg < threshold] = 0

    return cpImg


def score_binary_correlation( img1, img2, cmpArg  ):

    score = None
    h1=80
    h2=80
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
