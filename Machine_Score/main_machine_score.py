'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    03 Sep 2020
Description:    This is hopefully my final attempt at impliementing a pipeline scoring method.
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
import direct_image_compare as dc


# compare global variables

def test():
    print("MS: Hi!  You're in Matthew's SIMR module for all things machine scoring images")

def main(argList):

    # Prepare needed arguments
    if not hasattr( arg, 'new'): setattr( arg, 'new', False )

    if arg.printAll:

        print( "MS: Hi!  You're in Matthew's main comparison code" )
        gm.test()
        im.test()
        dc.test()
        arg.printArg()
    # End printAll

    if arg.simple:
        procSimple( 
                img1 = getattr( arg, 'img1', None ), \
                img2 = getattr( arg, 'img2', None ), \
                printAll = arg.printAll, \
                )
    # End simple 

    elif arg.runDir != None:

        print("MS: WARNING: Indevelopment to use new param file")

        pipelineRun( \
                arg.runDir, \
                printAll = arg.printAll, \
                tLoc = getattr( arg, 'targetLoc', None), \
                newInfo= getattr( arg, 'newInfo', None ), \
                )


    # If given a target directory
    elif arg.targetDir != None:

        pipelineTarget( \
                arg.targetDir, \
                printAll = arg.printAll, \
                newInfo = getattr( arg, 'newInfo', False), \
                nProcs = int( arg.nProc ), \
                nRuns = getattr( arg, 'nRuns', None ),                
                )

        # process entire data directory
    elif arg.dataDir != None: 

        pipelineAllData( arg.dataDir, printAll = arg.printAll, newInfo=arg.new, nProcs=int(arg.nProc) )


    #  Option not chosen 
    else:
        print("Error: MS: Please specify a directory to work in")
        print("\t -simple -img1 /path/to/img1 -img2 /path/to/img2")
        print("\t -runDir /path/to/dir/ -targetLoc /path/to/target")
        print("\t -targetDir /path/to/dir/")
        print("\t -dataDir /path/to/dir/")
        return

# End main

def procSimple( img1 = None, img2 = None, printAll = True ):

    if img1 == None: 
        print('Error: Please specify image 1 location')
        print('\t-img1 path/to/img1.png')
        return

    if img2 == None:
        print('Error: Please specify image 2 location')
        print('\t-img2 path/to/img1.png')
        return

    if not path.exists( img1 ) and not path.exists( img2 ):
        print('Error: An image was not found')
        print('\timg1: %s - \'%s\'' % ( str( path.exists( arg.img1 ) ), arg.img1 ) )
        print('\timg2: %s - \'%s\'' % ( str( path.exists( arg.img2 ) ), arg.img2 ) )


    img1, img2 = getImages( arg.img1, arg.img2 )

    simpleScore( img1, img2, printAll = arg.printAll )

    sList, cList = allScores( img1, img2, printAll = arg.printAll )
    for i, score in enumerate( sList ):
        print( '%s - %s - %s' % ( i, sList[i], cList[i] ) )
    
    return score



def pipelineAllData( dataDir, printAll = False, nProcs=1, newInfo=False ):

    sdssDirs = listdir( dataDir )
    nDirs = len( sdssDirs )

    for i,sDir in enumerate(sdssDirs):
        if printAll: print("CM: ***** TARGET DIR %d / %d *****" % ( i, nDirs ) )
        sdssDir = dataDir + sDir + '/'
        pipelineSdss( sdssDir, printAll = False, newInfo=newInfo, nProcs=nProcs )


    #print("TEST: Found %d dirs" % len( sdssDirs ) )


 
def pipelineTarget( tDir, printAll = False, nRuns=None, nProcs=1, newInfo=False ):

    if tDir[-1] != '/': tDir += '/'

    if printAll: 
        print("CM: pipelineTarget:")
        print("\t- printAll: %s" % printAll)
        print("\t- nRuns: %s" % nRuns)
        print("\t- nProcs: %s" % nProcs)
        print("\t- newInfo: %s" % newInfo)

    tInfo = im.target_info_class( targetDir = tDir, printAll = printAll, newInfo=newInfo )

    if tInfo.status == False:
        print("BAD!")
        return False

    gDir = tInfo.zooModelDir
    runDirs = listdir( gDir )
    runDirs.sort()

    if len( runDirs ) == 0:
        print("CM: WARNING: No runs directories found in sdssDir")
        print("\t- sdssDir: %s" % tDir)
        return False
    
    # Find target images
    sInfoContents = listdir( tInfo.infoDir )

    print("CM: pipelineTarget(): Hard coded to use 'target_zoo.png'")
    tImgLoc = tInfo.infoDir + 'target_zoo.png'

    if not path.exists( tImgLoc ):
        print("CM: pipelineTarget(): Target image location not found")
        print('\t- tLoc: %s' % tImgLoc)
        return False

    try:
        tImg = cv2.imread( tImgLoc, 0 )
        if printAll: print("CM: SDSS: read target img")
        tName = tImgLoc.split('/')[-1].split('.')[0]
        if printAll: print("CM: SDSS: Target name: %s" % tName)
    except:
        print("CM: ERROR: Failed to read target img")
        return False


    if nRuns != None:
        nRuns = int(nRuns)
        runDirs = runDirs[:nRuns] 
        print( "CM: TESTING: ", len( runDirs ), runDirs )

    # if single core
    if nProcs == 1:

        for i,run in enumerate(runDirs):
            if printAll: print('CM: runs: ', i, end='\r')
            rDir = gDir + run + '/'
            pipelineRun( rDir, tImg=tImg, tName=tName, printAll=False, newInfo=newInfo )
        if printAll: print('')

    # If running in parallel
    else:
 
        print("CM: pipelineTarget(): Using parallel processing!") 
        pClass = gm.ppClass( nProcs, printProg=printAll )

        runArgList = []

        for i,run in enumerate(runDirs):
            if printAll: print('CM: runs: ', i, end='\r')
            rDir = gDir + run + '/'
            
            runArg = dict( rDir=rDir, tImg=tImg, tName=tName, printAll=False, newInfo=newInfo )
            #runArg = dict( rDir=rDir, tImg=tImg, tName=tName, printAll=False, newInfo=newInfo )
            runArgList.append( runArg )

            #pipelineRun( rDir, tImg=tImg, tName=tName, printAll=False, newInfo=newInfo )

        pClass.loadQueue( pipelineRun, runArgList )

        pClass.runCores()

    tInfo.gatherRunInfos()
    tInfo.saveInfoFile()

# End processing sdss dir


# Create scores and comparison scores for run. 
def pipelineRun( \
        printBase=True, printAll=False, \
        runDir = None, rInfo = None, pClass = None, \
        tLoc=None, tImg=None, \
        imgLoc = None, mImg = None, \
        ):

    if printBase: 
        print("MS: pipelineRun:")

    if printAll: 
        print("\t - printBase: %s" % printBase )
        print("\t - printAll: %s" % printAll )
        print("\t - rInfo: %s" % type(rInfo) )
        print("\t - tLoc: %s" % tLoc )
        print("\t - tImg: %s" % type(tImg) )
        print("\t - imgLoc: %s" % imgLoc )
        print("\t - mImg: %s" % type(mImg) )

    if rInfo == None:
        rInfo = im.run_info_class( runDir=runDir, printAll=printAll, )

    # Check if successfully read info data
    if rInfo.status == False or pClass.status == False:
        if printBase:
            print("MS: Error: pipelineRun: Base status")
            print("\t- rInfo: ", rInfo.status)
            print("\t- pClass: ", pClass.status)
        return None

    # Check if score exists
    score = rInfo.getScore( pClass.get('name') ) 

    if score != None:
        if printBase: print("MS: Returning score")
        return score

    # Get target image
    scoreType = pClass.get('scoreType',None)
    if scoreType == None:
        print("MS: run: cmpTyscoreTypepe == None")
        return 

    elif scoreType == 'perturbation':
        tLoc = rInfo.findImgFile( pClass.get('imgArg')['name'], initImg=True )

    # Check if target image was given
    if tLoc == None and type( tImg ) == type( None ):
        if printBase:
            print("MS: ERROR: pipelineRun: ")
            print("\t- No target image given.")
        return None
    
    # Check if model image was given
    if imgLoc == None and type( mImg ) == type( None ):
        
        imgName = pClass.get( 'imgArg' ).get('name')
        imgLoc = rInfo.findImgFile( imgName )

        if imgLoc == None:
            print("MS: Failed to get image")
            return None

    # Open Target image
    if type(tImg) == type(None):
        
        if printAll: print("\t- Getting target image...")

        # Check if target image exists
        if not path.exists( tLoc ):
            print("Error: MS: Target image does not exist at: %s" % tLoc)
            return None

        tImg = getImg( tLoc )

        if type( tImg ) == type( None ):
            print("Error: MS: Target image error. %s" % tLoc)
            return None


    # Should have target image by now
    if printBase: print("\t - Target image good.")

    # Open model image 
    if type( mImg ) == type( None ):

        mImg = getImg( imgLoc )
        if type( mImg ) == type( None ):
            if printBase: print("Error: MS: Failed to read image: %s" % imgLoc)
            return None

    if printBase: print("\t - Model image good.")

    score = dc.createScore( tImg, mImg, cmpMethod=pClass.get('cmpArg')['cmpMethod'] )

    newScore = rInfo.addScore( name = pClass.get('name'), score=score )

    rInfo.saveInfoFile()
    return score

# end processing run dir

def getImg( imgLoc, printAll = False ):

    if not path.exists( imgLoc ):
        if printAll:
            print("MS: WARNING: image not found at path.")
            print("\t- %s" % imgLoc)
        return None

    img = cv2.imread( imgLoc, 0 ) 
    return img

# End get image


def allScores( img1, img2, printAll = False ):

    if img1.shape != img2.shape:
        img2 = img2.reshape( img1.shape )

    # Test if simple scoring function is working
    sList, cList = dc.allScores( img1, img2, printAll = printAll )

    return sList, cList


# simple score run.  Mostly for trouble shooting purposes
def simpleScore( img1, img2, printAll = False):

    score = None

    # Test if simple scoring function is working
    score = dc.score_absolute_difference( img1, img2, simple=True )

    if score == None:
        print("WARNING: MS: \n\t'None' score returned")

    if printAll:
        print("MS: Got simple Score!: %s" % str( score ) )

    # Test if complex thing is working
    score, cJson = dc.score_absolute_difference( img1, img2, simple=False )

    if score == None:
        print("WARNING: MS: \n\t'None' score returned")

    if printAll:
        if cJson != None:
            print("MS: Got complex score working!: %s" % ( score ) )
            print( cJson )
        else:
            print("WARNING: MS: \n\tGot score, but with None info: %s" % score )

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
        print('MS: Found images')
        print('\t img1: %s' % img1Loc)
        print('\t img2: %s' % img2Loc)
        print('MS: Opening images')

    img1 = None
    img2 = None

    img1 = cv2.imread( img1Loc, 0 )
    img2 = cv2.imread( img2Loc, 0 )

    if type( img1 ) != type( None ) and type( img2 ) != type( None ):
        print('MS: Opened images')
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
