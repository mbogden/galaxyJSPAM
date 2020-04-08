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
import info_module as im
import machineScoreMethods as ms

gm.test()
im.test()
ms.test()

# compare global variables

def main(argList):

    if arg.printAll:
        print( "MC: Printing Main Compare input arguments")
        arg.printArg()

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
        print("MC: WARNING: sdss dir in prog.")

        if arg.printAll: print("MC: Processing sdds dir: %s" % arg.sdssDir )

        pipelineSdss( arg.sdssDir, printAll = arg.printAll )


# End main
 
def pipelineSdss( sDir, printAll = False, n=0, nProc=1 ):

    if printAll: print("CM: pipelineSdss:")

    iDir = sDir + 'information/'
    gDir = sDir + 'gen000/'
    pDir = sDir + 'plots/'

    if not path.exists( iDir ) or not path.exists( gDir ):
        print("ERROR: sdssDir:", sDir)
        return False 

    runDirs = listdir( gDir )
    runDirs.sort()

    if len( runDirs ) == 0:
        print("CM: WARNING: No runs directories found in sdssDir")
        print("\t- sdssDir: %s" % sDir)
        return False
    
    # Find target images
    sInfoContents = listdir( iDir )

    for s in sInfoContents:
        print("\tTEST: %s" % s )
    
    print("CM: SDSS: Hard coded to use 'target_zoo.png'")
    tImgLoc = iDir + 'target_zoo.png'

    if not path.exists( tImgLoc ):
        print("CM: SDSS: Target image location not found")
        print('\t- tLoc: %s' % tImgLoc)
        return False

    try:
        tImg = cv2.imread( tImgLoc, 0 )
        if printAll: print("CM: SDSS: read target img")
    except:
        print("CM: ERROR: Failed to read target img")
        return False

    print("CM: TESTING: sdss n set to 10")
    #n = 10
    if n != 0:
        runDirs = runDirs[:n] 
        print( "CM: TESTING: ", len( runDirs ), runDirs )

    # if single core
    if nProc == 1:

        for i,run in enumerate(runDirs):
            print('TEST: ', i)
            rDir = gDir + run + '/'
            pipelineRun( rDir, tImg=tImg, printAll=False, )


# End processing sdss dir


# Create perturbedness and comparison scores for run. 
def pipelineRun( rDir, printAll=False, rmInfo=False, tLoc=None, tImg=None, tName=None, paramName=None, overwriteScores=False ):

    if printAll: print("MC: In pipeline run")

    # Create paths to expected files and directories
    modelDir = rDir + 'model_images/'
    ptsDir   = rDir + 'particle_files/'
    infoLoc  = rDir + 'info.json'

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
    elif tLoc == None and type(tImg) == type(None):
        print("Error: MC: No target data given for run: %s" % arg.runDir)
        return False

    # Read/create run Info file
    if printAll: print("\t- Initizing info class for run.")
    
    # remove info file.  mostly for initial creation of code and troubleshooting.
    if rmInfo and path.exists( infoLoc ):
        from os import system

        if printAll: print("\t- Removing info.json from run...")
        system( 'rm %s' % infoLoc )

    rInfo = im.run_info_class( infoLoc=infoLoc, runDir=rDir, printAll=printAll )

    # Check if successfully read info data
    if rInfo.status == False:
        print("Error: MC: Failed to get info from: \n\t- %s" % infoLoc)
        return False

    if printAll: print("MC: Obtained run info.")

    mDictList = rInfo.rDict.get( 'model_images', None )
    iDictList = rInfo.rDict.get( 'misc_images', None )
    pScores = rInfo.rDict.get( 'perturbedness', None )
    mScores = rInfo.rDict.get( 'machine_scores', None )

    if mDictList == None or iDictList == None or pScores == None or mScores == None:
        print("Error: MC: info json not complete")
        print("\t- ", mDictList)
        print("\t- ", iDictList)
        print("\t- ", pScores)
        print("\t- ", mScores)
        return False

    # Can't do anything if list is empty
    if len( mDictList ) == 0:
        print("WARNING: MC: Model Image list empty")
        return False
       # Check if machine scores need to be populated

    # simple check if scores exist
    compareList = ms.getScoreFunctions()
    nScores = len( compareList )

    doPerturbedness = True
    doMachineScores = True

    # check for existing machine scores
    if not overwriteScores and len(mScores) >= nScores*len( mDictList ):
        if printAll: print("MC: Machine fitness scores already found")
        doMachineScores = False
    else:
        if printAll: print("MC: Creating machine scores")

    # check for existing perturbness overwrite
    if not overwriteScores and len(pScores) >= nScores*len( mDictList ):
        if printAll: print("MC: Perturbedness scores already found")
        doPerturbedness = False
    else:
        if printAll: print("MC: Creating perturbedness scores")

    if not doPerturbedness and not doMachineScores:
        if printAll: print("MC: No scores needed")
        return False

    # Open Target image
    if type(tImg) == type(None) and tLoc != None:
        
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

        tName = tLoc.split('/')[-1]

    # Should have target image by now
    if printAll: print("\t- Target image good.")

    # filter by image parameter name if needed
    if paramName != None:
        mDictList = [ mD for mD in mDictList if mD['image_parameter_name'] == paramName ]
        if len( mDictList ) == 0:
            print("MC: WARNING: No images with paramName: %s" % paramName )

    # Go through model images and create scores
    for mD in mDictList:

        mName = mD.get( 'image_name', None )
        mParam = mD.get( 'image_parameter_name', None )

        if mName == None or mParam == None:
            print("WARNING: MC: Bad image name found")
            print("\t- ", mD)
            continue

        mLoc = rDir + 'model_images/' + mName
        mImg = getImg( mLoc )

        # Create and save machine scores
        if doMachineScores:

            if printAll: print("\t- Creating machine scores for %s" % mName)
            sList, cList = allScores( mImg, tImg, printAll = printAll )

            # append image names to machine score dict
            for c in cList:
                c['model_name'] = mName
                c['target_name'] = tName

            new_mScores = rInfo.appendList( 'machine_scores', cList )

            if new_mScores != None:
                mScores = new_mScores
            else:
                if printAll: print("MC: WARNING: Failed creating perturbedness for %s" % mName)


        # create and saving perturbedness scores
        if doPerturbedness:

            if printAll: print("\t- Creating perturbedness for %s" % mName)

            # Find matching initial image
            iL = [ iD['image_name'] for iD in iDictList if iD['image_parameter_name'] == mParam ]
            if len( iL ) == 0:
                print("MC: WARNING: Found no matching initial images for model image")
                print("\t- %s" % mD)
                continue

            iName = iL[0]
            iLoc = rDir + 'model_images/' + iName
            iImg = getImg( iLoc )

            sList, cList = allScores( mImg, iImg, printAll = printAll )

            # append image names to perturbedness dict
            for c in cList:
                c['model_name'] = mName

            new_pScores = rInfo.appendList( 'perturbedness', cList )

            if new_pScores != None:
                pScores = new_pScores
            else:
                if printAll: print("MC: WARNING: Failed creating perturbedness for %s" % mName)

    # Done with perturbedness and machine scores

    rInfo.saveInfoFile()

# end processing run dir

def getImg( imgLoc ):

    if not path.exists( imgLoc ):
        print("MC: WARNING: image not found at path.")
        print("\t- %s" % imgLoc)
        return None

    img = cv2.imread( imgLoc, 0 ) 
    return img
# End get image


def allScores( img1, img2, printAll = False ):

    if img1.shape != img2.shape:
        img2 = img2.reshape( img1.shape )

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
# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
