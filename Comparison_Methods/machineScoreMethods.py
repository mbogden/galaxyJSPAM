'''
     Author:    Matthew Ogden
    Created:    31 Oct 2019
    Altered:    21 Feb 2020
Description:    Created so comparison methods are simply maintained
'''

import cv2
import json
import numpy as np

def test():
    print("MS: Hi!  Inside machineScoreMethods.py")

def score_absolute_difference( img1, img2, simple=True ):

    score = None

    # adjust so mean brightness matches

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score

    # if simply return score
    if simple:
        return score

    # if complex return 
    cInfo = None

    cInfo = {
        'score' : score,
        'comparison_name': 'absolute_difference'
    }

    cJson = json.dumps( cInfo )

    return score, cJson
# End absDiff

def score_absolute_difference_lower_scale( img1, img2, l = 0.9, simple=True  ):

    score = None

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score

    # arbitrary adjustment to make score more human readable 
    score = (score - l) / (1-l)
    if score < 0.0: score = 0.0

    if simple: return score

    # if complex return 
    cInfo = None

    cInfo = {
        'score' : score,
        'comparison_name': 'absolute_difference_lower_scale', 
        'comparison_info': {
            'lower_scale' : l
        }
    }

    cJson = json.dumps( cInfo )

    return score, cJson

# end createMScores()

# All basic scoring methods

def score_overlap_fraction( img1, img2, h=100, simple=True ):

    score = None

    i1 = np.copy( img1 )
    i2 = np.copy( img2 )

    i1[ i1 <  h ] = 0
    i1[ i1 >= h ] = 1

    i2[ i2 <  h ] = 0
    i2[ i2 >= h ] = 1

    bImg = i1 + i2
    bImg[ bImg <  2 ] = 0
    bImg[ bImg >= 2 ] = 1

    x = np.sum( i1 )
    y = np.sum( i2 )
    z = np.sum( bImg )

    score = ( z / ( x + y - 1.0*z ) )
    
    if simple:  return score

    # if complex return 
    cInfo = None

    cInfo = {
        'score' : score,
        'comparison_name': 'overlap_fraction',
        'comparison_info': {
            "h" : h
        }
    }

    cJson = json.dumps( cInfo )
    return score, cJson


def score_correlation( img1, img2, simple=True  ):

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

    if simple: return score

    # if complex return 
    cInfo = None

    cInfo = {
        'score' : score,
        'comparison_name': 'correlation'
    }

    cJson = json.dumps( cInfo )
    return score, cJson

def binImg( imgIn, threshold ):

    cpImg = np.copy( imgIn )
    cpImg[ cpImg >= threshold] = 255
    cpImg[ cpImg < threshold] = 0

    return cpImg


def score_binary_correlation( img1, img2, t1=155, t2=None, bin1=True, bin2=True, simple=True  ):

    score = None
    
    if bin1:
        img1 = binImg( img1, t1 )
    
    if bin2:
        if t2 == None: t2 = t1
        img2 = binImg( img2, t2 )

    score = score_correlation( img1, img2 )

    if simple: return score

    # if complex return 
    cInfo = None
    cInfo = {
        'score' : score,
        'comparison_name': 'binary_correlation',
        'comparison_info': {
            't1' : "%s" % t1, 
            't2' : "%s" % t2, 
        }
    }
    cJson = json.dumps( cInfo )

    return score, cJson

# end createBinaryCorrelation()

def getScoreFunctions():

    import inspect, sys
    scoreFunctions = [ ( name, obj ) for name,obj in inspect.getmembers(sys.modules[__name__]) \
                        if ( inspect.isfunction(obj) and name.split('_')[0] == 'score' ) ]
    
    return scoreFunctions


def allScores( img1, img2, printAll = False ):

    fList = getScoreFunctions()

    sList = []
    cList = []

    for i,l in enumerate(fList):

        fName, func = l

        score, cJson = func( img1, img2, simple=False )

        sList.append( score )
        cList.append( cJson )

    return sList, cList 



# Run main after declaring functions
if __name__=='__main__':

    test()

    import inspect, sys
    scoreFunctions = [ ( name, obj ) for name,obj in inspect.getmembers(sys.modules[__name__]) \
                        if ( inspect.isfunction(obj) and name.split('_')[0] == 'score' ) ]
    
    #print( scoreFunctions )

    print("Score Functions Found")
    for n,f in scoreFunctions:
        print('\t-',n)
