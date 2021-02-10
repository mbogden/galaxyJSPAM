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



# Create scores and comparison scores for run. 
def MS_Run( \
        printBase=True, printAll=False, \
        runDir = None, rInfo = None, \
        params = None, arg = None, \
        ):

    if printBase: 
        print("MS: main_run:")

    if rInfo == None:
        rInfo = im.run_info_class( runDir=runDir, printAll=printAll, )

    # Check if successfully read info data
    if rInfo.status == False:
        if printBase:
            print("MS: Error: pipelineRun: Base status")
            print("\t- rInfo: ", rInfo.status)
        return None

    # Check if params were given
    if params == None:
        if printBase:
            print("MS: Error: pipelineRun: Bad score parameters")
            print("\t- params: ", params)
        return None

    # Check if params were given
    if arg == None:
        if printBase:
            print("MS: Error: pipelineRun: Bad argument input")
            print("\t- arg type: ", type(arg) )
        return None

    # Assume group of parameters and loop through.
    
    for pKey in params:
        
        # Check if score exists
        score = rInfo.getScore( pKey )
        
        # If score exists, move to next score parameter
        if score != None and not arg.get('overWrite',False):
            continue
        
        # Grab parameter and score type
        param = params[pKey]
        scoreType = param['scoreType']
        
        # Call function with correct score type
        if scoreType == 'target':
            target_image_compare(rInfo, param, arg) 
        else:
            print("WARNING: MS: Run")
            print("Score Type of '%s' not yet implemented"%(scoreType))
    
# end processing run dir



# compares a target and image with given score parameters
def target_image_compare(rInfo, param, args):
    
    printBase = args.printBase
    
    # GET Target Image
    tName = param['targetName'] 
    tLoc = None
    tImg = None
    tLink = rInfo.get('link_tInfo')
    
    # Create dict for storing target and model images if needed
    if rInfo.get('targetImg') == None:
        rInfo.targetImg = {}
        
    if rInfo.get('modelImg') == None:
        rInfo.modelImg = {}
        
    # Check if in rInfo has target Image already
    if type( rInfo.targetImg.get(tName) ) != type(None):
        tImg = rInfo.targetImg.get(tName)
    
    # Check other source
    if type(tLoc) == type(None) and type(tLink) == type(None):
        if printBase:
                print("MS: Error: target_image_compare: no target image or link given")
        return None
    
    # If given tLink
    if type(tLoc) == type(None):
        tLoc = tLink.findTargetImage(tName)
        tImg = gm.readImg(tLoc)
    
        # Finally, should have tImg.  Leave if not
        if type(tImg) != None:
            rInfo.targetImg[tName] = tImg
        else:
            if printBase:
                print("MS: Error: target_image_compare: failed to load target image")
            return None
    
    # GET Model Image
    
    mName = param['imgArg']['name']
    mImg = None
    mloc = None
        
    # Check if in rInfo has model Image already
    if type( rInfo.modelImg.get(mName) ) != type(None):
        mImg = rInfo.modelImg.get(mName)
    
    # Else, have rInfo retrieve img location
    else:
        mLoc = rInfo.findImgLoc(mName)
        if gm.validPath(mLoc):
            mImg = gm.readImg(mLoc)
            rInfo.modelImg[mName] = mImg
        
    if type(mImg) == type(None): 
            if printBase: 
                print("MS: Error: target_image_compare: failed to load model image")
                gm.tabprint("model: "%mName)
            return None
    
    # Create score and add to rInfo
    
    score = dc.createScore( tImg, mImg, param['cmpArg'] )
    newScore = rInfo.addScore( name = param['name'], score=score )
    rInfo.saveInfoFile()
    
    return score


# TODO
def perturbation_image_compare(rInfo, param, args):
    print(param)
        
    tLoc = rInfo.findImgFile( pClass.get('imgArg')['name'], initImg=True )

    return None


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

# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
