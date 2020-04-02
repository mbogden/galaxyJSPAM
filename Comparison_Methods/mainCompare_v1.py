'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    21 Feb 2020
Description:    This is my new attempt at having a comparison program for implementing comparison methods.
'''

from os import path, listdir
from sys import exit, argv, path as sysPath

import numpy as np
import cv2
import pandas as pd

# For importing my useful support module
supportPath = path.abspath( path.join( __file__, "../../Support_Code/" ) )
sysPath.append( supportPath )

import general_module as gm
import score_module as sm
import machineScoreMethods as ms

gm.test()
sm.test()
ms.test()

# compare global variables

def main(argList):

    if arg.printAll:
        print( "MC: Printing Main Compare input arguments")
        arg.printAllArg()

    if arg.simple == False and arg.runDir == None and arg.sdssDir == None and arg.dataDir == None:
        print("Error: MC: Please specify a directory to work in")
        print("\t -simple -img1 /path/to/img1 -img2 /path/to/img2")
        print("\t -runDir /path/to/dir/ -targetLoc /path/to/target")
        print("\t -sdssDir /path/to/dir/")
        print("\t -dataDir /path/to/dir/")
        return

    if arg.simple:

        if not hasattr( arg, 'img1'):
            print('Error: Please specify image 1 location')
            print('\t-img1 path/to/img1.png')
            return

        if not hasattr( arg, 'img2'):
            print('Error: Please specify image 2 location')
            print('\t-img2 path/to/img1.png')
            return

        if not path.exists( arg.img1 ) and not path.exists( arg.img2 ):
            print('Error: An image was not found')
            print('\timg1: %s - \'%s\'' % ( str( path.exists( arg.img1 ) ), arg.img1 ) )
            print('\timg2: %s - \'%s\'' % ( str( path.exists( arg.img2 ) ), arg.img2 ) )


        img1, img2 = getImages( arg.img1, arg.img2 )

        simpleScore( img1, img2, printAll = arg.printAll )

        sList, cList = allScores( img1, img2, printAll = arg.printAll )
        for i, score in enumerate( sList ):
            print( '%s - %s - %s' % ( i, sList[i], cList[i] ) )
        
        return score


    # Only scoring a single run directory
    if arg.runDir != None:

        # Check if targetLoc was given in arguments
        if not hasattr( arg, 'targetLoc'):
            print("Error: Please specify a target image location for runDir.")
            print("\t -targetLoc /path/to/img.png.")
            return False

        pipelineRun( arg.runDir, tLoc = arg.targetLoc, printAll = arg.printAll )
        #pipelineRun( arg.runDir, arg.targetLoc, paramName='other', printAll=True )
        #pipelineRun( arg.runDir, arg.targetLoc, paramName='test' )

    elif arg.sdssDir != None:
        print("WARNING:\n\tSdss directory not working yet.  Currently working on runDir.")

# End main
 
# Create scores assuming you're working with current run directory
def pipelineRun( rDir, tLoc=None, tImg=None, paramName=None, overwriteScore=True, printAll=False ):

    if printAll: print("MC: In pipeline run")
       
    # Create paths to expected files and directories
    modelDir = rDir + 'model_images/'
    ptsDir   = rDir + 'particle_files/'
    infoLoc  = rDir + 'info.txt'
    scoreLoc = rDir + 'scores.json'

    mImgs = []  # List of images

    # Check if run directory exists
    if not path.exists( rDir ):
        print("Error: MC: Run directory does not exist at: %s" % arg.runDir)
        return False
     
    # Check model directory exists
    if not path.exists( modelDir ):
        print("Error: MC: runDir folders not found: %s" % rDir)
        return False

    # Check if target data was given
    elif tLoc == None and tImg == None:
        print("Error: MC: No target data given for run: %s" % arg.runDir)
        return False

    # Open Target image if only location is given
    elif tImg == None and tLoc != None:
        
        if printAll: print("\t- Getting target image...")

        # Check if target image exists
        if not path.exists( tLoc ):
            print("Error: MC: Target image does not exist at: %s" % tLoc)
            return False

        # Open image.  Catch exception if something happens.
        try:
            tImg = cv2.imread( tLoc, 0 )
        except:
            print("Error: MC: Failed to open target image: %s" % tLoc)
            return False

    # Should have target image tImg by now
    if printAll: print("\t- Target image good.")
   
    imgLocs = []

    if printAll: print("\t- Getting Model Images...")
    # Grab all models if no paramName given

    if paramName == None:
        allImgs = listdir( modelDir )
        imgLocs = [ modelDir + img  \
                    for img in allImgs \
                    if 'model.png' in img  and path.exists(modelDir + img) ]

    # Grab only models with matching name
    else:
        iLoc = modelDir + '%s_model.png' % paramName
        if path.exists( iLoc ):
            imgLocs.append( iLoc )
        else:
            print("Error: MC: Model image not found: %s" % iLoc)

    # Give warning if no model files found
    if len( imgLocs ) == 0:
        print("Error: MC: No model image(s) found: %s" % modelDir)
        return
    
    if printAll:
        print("\t- Model Images found")
        for iLoc in imgLocs: print('\t\t-',iLoc)

    # Read/create run Score file
    if printAll: print("\t- Initizing score class for run...")
    sObj = sm.run_score_class( scoreLoc=scoreLoc, infoLoc=infoLoc, printAll=True )

    # Check if successfully read score data
    if sObj.status == False:
        print("Error: Failed to get score data from %s" % scoreLoc)
        return False

    if printAll: print("MC: Got run score data.")

    # Previous Image list in score file
    pImgList =  

    if pImgList == None:
        pImgList = []
        sObj.rDict['model_images'] = pImgList

    print( pImgList )

    for iLoc in imgLocs:
        iName = iLoc.split('/')[-1]
        print(iName)


# end processing run dir


def allScores( img1, img2, printAll = False ):

    # Test if simple scoring function is working
    sList, cList = ms.allScores( img1, img2, printAll = printAll )

    return sList, cList


# simple score run.  Mostly for trouble shooting purposes
def simpleScore( img1, img2, printAll = False):

    score = None

    # Test if simple scoring function is working
    score = ms.score_absolute_difference( img1, img2, simple=True )

    if score == None:
        print("WARNING: MC: \n\t'None' score returned")

    if printAll:
        print("MC: Got simple Score!: %s" % str( score ) )

    # Test if complex thing is working
    score, cJson = ms.score_absolute_difference( img1, img2, simple=False )

    if score == None:
        print("WARNING: MC: \n\t'None' score returned")

    if printAll:
        if cJson != None:
            print("MC: Got complex score working!: %s" % ( score ) )
            print( cJson )
        else:
            print("WARNING: MC: \n\tGot score, but with None info: %s" % score )

    return score

# Getting images for simple score comparison
def getImages( img1Loc, img2Loc, printAll = False ):

    if not path.exists( img1Loc ):
        print("Error:")
        print("\timage path 'img1' does not exist: '%s'" % img1Loc)
        return None

    if not path.exists( img2Loc ):
        print("Error:")
        print("\timage path 'img2' does not exist: '%s'" % img2Loc)
        return None

    if printAll:
        print('MC: Found images')
        print('\t img1: %s' % img1Loc)
        print('\t img2: %s' % img2Loc)
        print('MC: Opening images')

    img1 = None
    img2 = None

    img1 = cv2.imread( img1Loc, 0 )
    img2 = cv2.imread( img2Loc, 0 )

    if type( img1 ) != type( None ) and type( img2 ) != type( None ):
        print('MC: Opened images')
    else:
        if type( img1 ) != type( None ):
            print('\tFailed to open img1: %s' % img1Loc)
        else:
            print('\tFailed to open img2: %s' % img2Loc)

    if img1.shape != img2.shape:
        print("WARNING: CM: simple:\n\tImagse do not match shape")
        img2 = np.reshape( img2, img1.shape )

    return img1, img2

def procSdss( sDir ):
    iDir = sDir + 'information/'
    gDir = sDir + 'gen000/'
    pDir = sDir + 'plots/'
    scoreDir = sDir + 'scores/'

    if not path.exists( iDir ) or not path.exists( gDir ):
        print("Error sdssDir:", sDir)
        return 

    runDirs = listdir( gDir )
    runDirs.sort()

    for run in runDirs:
        rDir = gDir + run + '/'
        procRun( rDir )
# End processing sdss dir

# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
