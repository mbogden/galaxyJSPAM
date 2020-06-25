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
import machine_score_methods as ms


# compare global variables

def test():
    print("MC: You are in main_compare.py")

def main(argList):

    # Prepare needed arguments
    if not hasattr( arg, 'new'): setattr( arg, 'new', False )

    if arg.printAll:

        print( "MC: Hi!  You're in Matthew's main comparison code" )
        gm.test()
        im.test()
        ms.test()
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

        pipelineRun( \
                arg.runDir, \
                printAll = arg.printAll, \
                tLoc = getattr( arg, 'targetLoc', None), \
                rmInfo= getattr( arg, 'rmInfo', None ) \
                )



    # If given a target directory
    elif arg.targetDir != None:

        pipelineTarget( \
                arg.targetDir, \
                printAll = arg.printAll, \
                rmInfo = getattr( arg, 'rmInfo', False), \
                nProcs = int( arg.nProc ), \
                nRuns = getattr( arg, 'nRuns', None ),                
                )

        # process entire data directory
    elif arg.dataDir != None: 

        pipelineAllData( arg.dataDir, printAll = arg.printAll, rmInfo=arg.new, nProcs=int(arg.nProc) )


    #  Option not chosen 
    else:
        print("Error: MC: Please specify a directory to work in")
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



def pipelineAllData( dataDir, printAll = False, nProcs=1, rmInfo=False ):

    sdssDirs = listdir( dataDir )
    nDirs = len( sdssDirs )

    for i,sDir in enumerate(sdssDirs):
        if printAll: print("CM: ***** TARGET DIR %d / %d *****" % ( i, nDirs ) )
        sdssDir = dataDir + sDir + '/'
        pipelineSdss( sdssDir, printAll = False, rmInfo=rmInfo, nProcs=nProcs )


    #print("TEST: Found %d dirs" % len( sdssDirs ) )


 
def pipelineTarget( tDir, printAll = False, nRuns=None, nProcs=1, rmInfo=False ):

    if tDir[-1] != '/': tDir += '/'

    if printAll: 
        print("CM: pipelineTarget:")
        print("\t- printAll: %s" % printAll)
        print("\t- nRuns: %s" % nRuns)
        print("\t- nProcs: %s" % nProcs)
        print("\t- rmInfo: %s" % rmInfo)

    tInfo = im.target_info_class( targetDir = tDir, printAll = printAll, rmInfo=rmInfo )

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
            pipelineRun( rDir, tImg=tImg, tName=tName, printAll=False, rmInfo=rmInfo )
        if printAll: print('')

    # If running in parallel
    else:
 
        print("CM: pipelineTarget(): Using parallel processing!") 
        pClass = gm.ppClass( nProcs, printProg=printAll )

        runArgList = []

        for i,run in enumerate(runDirs):
            if printAll: print('CM: runs: ', i, end='\r')
            rDir = gDir + run + '/'
            
            runArg = dict( rDir=rDir, tImg=tImg, tName=tName, printAll=False, rmInfo=rmInfo )
            #runArg = dict( rDir=rDir, tImg=tImg, tName=tName, printAll=False, rmInfo=rmInfo )
            runArgList.append( runArg )

            #pipelineRun( rDir, tImg=tImg, tName=tName, printAll=False, rmInfo=rmInfo )

        pClass.loadQueue( pipelineRun, runArgList )

        pClass.runCores()


    print("CM: pipelineTarget(): Not updating tInfo...")
    #tInfo.gatherRunInfos()
    #tInfo.saveInfoFile()

# End processing sdss dir


# Create perturbation and comparison scores for run. 
def pipelineRun( rDir, printAll=False, rmInfo=False, tLoc=None, tImg=None, tName=None, paramName=None, overwriteScores=False ):

    if printAll: 
        print("MC: pipelineRun:")
        print("\t- printAll: %s" % printAll)
        print("\t- rmInfo: %s" % rmInfo)
        print("\t- tLoc: %s" % tLoc)
        print("\t- tImg: %s" % tImg)
        print("\t- tName: %s" % tName)
        print("\t- paramName: %s" % paramName)
        print("\t- overwriteScores: %s" % overwriteScores)

    mImgs = []  # List of images

    # Check if target data was given
    if tLoc == None and type(tImg) == type(None):
        print("ERROR: MC: pipelineRun: ")
        print("\t- No target image. runDir: %s" % arg.runDir)
        return False


    rInfo = im.run_info_class( runDir=rDir, printAll=printAll, rmInfo=rmInfo )

    # Check if successfully read info data
    if rInfo.status == False:
        print("Error: MC: pipelineRun: ")
        print("\t- Bad status from run_info_class.")
        print("\t- runDir: %s" % runDir)
        return False
    else: 
        if printAll: print("MC: Obtained run info.")

    # Get info from run
    mDictList = rInfo.rDict.get( 'model_images', None )
    iDictList = rInfo.rDict.get( 'misc_images', None )
    pScores = rInfo.rDict.get( 'perturbation', None )
    mScores = rInfo.rDict.get( 'machine_scores', None )
    iScores = rInfo.rDict.get( 'initial_bias', None )


    # Check list for needed run information
    checkList = [ 'model_images', 'init_images' ]
    #rInfo.checkList( checkList )
    if printAll:
        print("MC: pipelienRun: WARNING:")
        print("\t- Consider placing checkList, overwrite, and image retreival inside info module.")

    if mDictList == None or iDictList == None:
        print("Error: MC: info json not complete")
        print("\t- ", mDictList)
        print("\t- ", iDictList)
        rInfo.printInfo()
        rInfo.updateInfo()
        rInfo.printInfo()

        mDictList = rInfo.rDict.get( 'model_images', None )
        iDictList = rInfo.rDict.get( 'misc_images', None )
        if mDictList == None or iDictList == None:

           return False
         

    if pScores == None or mScores == None or iScores == None:
        print("BLAH!")
        return False

    # Can't do anything if list is empty
    if len( mDictList ) == 0:
        print("WARNING: MC: Model Image list empty")
        return False
       # Check if machine scores need to be populated

    # simple check if scores exist
    compareList = ms.getScoreFunctions()
    nScores = len( compareList )

    doMachineScores = True
    doMetaScores = True

    if printAll: print("MC: pipelineRun(): Currently disabling overwrite settings")

    '''
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
        if printAll: print("MC: Creating perturbation scores")

    if not doPerturbedness and not doMachineScores:
        if printAll: print("MC: No scores needed")
        return False
    '''

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

        mName = mD
        mParam = mDictList[mD]['image_parameter_name']

        if mName == None or mParam == None:
            print("MC: pipelineRun: WARNING: ")
            print("\t- Bad image name found")
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

            new_mScores = rInfo.appendScores( keyName='machine_scores', addList=cList )

            if new_mScores != None:
                mScores = new_mScores
            else:
                if printAll: print("MC: WARNING: Failed creating perturbation for %s" % mName)


        # create and saving perturbation scores
        if doMetaScores:

            if printAll: print("\t- Creating perturbation and initial bias for %s" % mName)

            # Find matching initial image
            '''
            iL = [ iD['image_name'] for iD in iDictList if iD['image_parameter_name'] == mParam ]
            if len( iL ) == 0:
                print("MC: WARNING: Found no matching initial images for model image")
                continue
            '''

            iLoc = rDir + 'misc_images/' + mName.replace('model','init')
            iImg = getImg( iLoc )
            if type(iImg) == type(None):
                print("BAD")
                continue

            psList, pcList = allScores( mImg, iImg, printAll = printAll )
            isList, icList = allScores( tImg, iImg, printAll = printAll )

            # append image names to perturbation dict
            for i,c in enumerate(pcList):
                pcList[i]['model_name'] = mName
                icList[i]['model_name'] = mName
                icList[i]['target_name'] = tName

            new_pScores = rInfo.appendScores( keyName='perturbation', addList=pcList )
            new_iScores = rInfo.appendScores( keyName='initial_bias' , addList=icList )

            if new_pScores == None:
                 print("MC: WARNING: Failed creating perturbation for %s" % mName)
                 print(pcList)
                 print(new_pScores)
                 print(rInfo.rDict['perturbation'])

            if new_iScores == None:
                 print("MC: WARNING: Failed creating initial_bias for %s" % mName)
                 print(icList)
                 print(new_iScores)
                 print(rInfo.rDict['initial_bias'])


    # Done with perturbation and machine scores

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
