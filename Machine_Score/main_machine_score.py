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

def test():
    print("MS: Hi!  You're in Matthew's SIMR module for all things machine scoring images")

# global variables
testFunc = None

# When creating and testing new functions from outside of module (Ex. Jupyter)
def set_test_score( inFuncLink ):
    global testFunc
    testFunc = inFuncLink

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
        printBase=None, printAll=None, \
        runDir = None, rInfo = None, \
        params = None, arg = gm.inArgClass(), \
        ):
    
    if printBase == None:
        printBase = arg.printBase
        
    if printAll == None:
        printAll = arg.printAll
    
    if printAll:
        printBase = True

    if printBase: 
        print("MS: Run:")
        
    if rInfo == None and runDir == None and arg.get('rInfo') == None:
        if printBase:
            print("MS: Error: MS_Run: No run info given")
        return None

    elif runDir != None and rInfo == None:
        rInfo = im.run_info_class( runDir=runDir, printAll=printAll, )
    
    elif arg.get('rInfo') != None and rInfo == None:
        rInfo = arg.rInfo

    # Check if successfully read info data
    if rInfo.status == False:
        if printBase:
            print("MS: Error: MS_Run: run info status bad")
            print("\t- rInfo: ", rInfo.status)
        return None

    # Check if params were given
    if params == None and arg.get('scoreParams',None) == None:
        if printBase:
            print("MS: Error: MS_Run: Score parameters not given")
        return None

    # Check if params were given
    if params == None:
        params = arg.get('scoreParams',None)
        if type(params) != type({'dict':'dict'}):
            if printBase:
                print("MS: Error: MS_Run: Bad score parameters")
                gm.tabprint('param_type: %s'%type(params))
            return None
    

    # Assume group of parameters and loop through.    
    for pKey in params:
        
        # Check if score exists
        score = rInfo.getScore( pKey )
        if printAll: print("MS: scoreName: %s"%pKey)
        
        # If score exists, move to next score parameter
        if score != None and not arg.get('overWrite',False):
            continue
        
        # Grab parameter and score type
        param = params[pKey]
        
        # Check if score parameter if valid
        scoreType = param.get('scoreType',None)
        if scoreType == None:
            if printBase: 
                print("WARNING: MS: Score Type invalid: %s"%param.get('name',None))
            continue
        
        # Check if compare arguments are there. 
        if param.get('cmpArg',None) == None:
            if printBase: 
                print("WARNING: MS: compare Args invalid: %s"%param.get('name',None))
            continue
        
        # Call function with correct score type
        if scoreType == 'target':
            target_image_compare( rInfo, param, arg )
        elif scoreType == 'perturbation':
            perturbation_compare( rInfo, param, arg )
        else:
            print("WARNING: MS: Run")
            print("Score Type of '%s' not yet implemented"%(scoreType))
    
# end processing run dir


# compares a target and image with given score parameters
def target_image_compare( rInfo, param, args ):
    
    printBase = args.printBase
    printAll = args.printAll
    
    # Base info
    pName = param['name']
    tName = param['targetName'] 
    mName = param['imgArg']['name']
    
    # GET TARGET IMAGE
    tLoc = args.get('targetLoc',None)
    tImg = None
    tLink = rInfo.get('tInfo')
    
    if printBase: print('MS: target_image_compare: %s'%pName)
    
    if printAll:
        im.tabprint(' paramName: %s'%pName)
        im.tabprint(' modelName: %s'%mName)
        im.tabprint('targetName: %s'%tName)
    
    # Exit if invalid request
    if type(tLoc) == type(None) and type(tLink) == type(None):
        if printBase:
                print("MS: Error: target_image_compare: no target image or link given")
        return None
        
    # Check if image location given
    if gm.validPath(tLoc, printWarning=printBase):
        tImg = gm.readImg(tLoc)
        
    # If given tInfo link, have tInfo search and get image.
    else:        
        tImg = tLink.getTargetImage(tName)
    
    # Finally, should have tImg.  Leave if not
    if type(tImg) == None:
        if printBase:
            print("MS: Error: target_image_compare: failed to load target image")
        return None
    elif printAll: gm.tabprint("Read target image")
    
    # GET MODEL IMAGE    
    mImg = None
    mloc = None
    
    # Create dict for storing target and model images if needed        
    if rInfo.get('modelImg') == None:
        rInfo.modelImg = {}
        
    # Check if in rInfo has model Image already
    if type( rInfo.modelImg.get(mName) ) != type(None):
        mImg = rInfo.modelImg.get(mName)
    
    # Else, have rInfo retrieve img location
    else:
        mImg = rInfo.getModelImage(mName)
        rInfo.modelImg[mName] = mImg
        
    if type(mImg) == type(None): 
        if printBase: 
            print("MS: Error: target_image_compare: failed to load model image")
            gm.tabprint("runId: %s"%rInfo.get('run_id'))
            gm.tabprint("model: %s"%mName)
        return None
        
    elif printAll: print("MS: run: Read model image")
    
    # Create score and add to rInfo
    
    score = dc.createScore( tImg, mImg, param['cmpArg'] )
    newScore = rInfo.addScore( name = param['name'], score=score )
    #rInfo.saveInfoFile()
    
    if printAll: print("MS: New Score!: %s - %f - %f" % (param['name'],score, newScore))
    
    return score


def perturbation_compare( rInfo, param, args ):
    
    printBase = args.printBase
    printAll = args.printAll
    
    # Base info
    pName = param['name']
    mName = param['imgArg']['name']
    
    if printBase: print('MS: perturbation_compare: %s'%pName)
    
    if printAll:
        im.tabprint(' paramName: %s'%pName)
        im.tabprint(' modelName: %s'%mName)
    
    # Exit if invalid request
    if type(pName) == type(None) and type(mName) == type(None):
        if printBase:
                print("MS: Error: target_image_compare: Bad param given")
        return None
    
    # GET MODEL IMAGE    
    mImg = None
    mLoc = None
    
    # Create dict for storing model images if needed later       
    if rInfo.get('modelImg') == None:
        rInfo.modelImg = {}
        
    # Check if in rInfo has model Image already
    if type( rInfo.modelImg.get(mName) ) != type(None):
        mImg = rInfo.modelImg.get(mName)
    
    # Else, have rInfo retrieve img location
    else:
        mImg = rInfo.getModelImage(mName)
        rInfo.modelImg[mName] = mImg
    
    # Check if valid image
    if type(mImg) == type(None): 
        if printBase: 
            print("MS: Error: target_image_compare: failed to load model image")
            gm.tabprint("runId: %s"%rInfo.get('run_id'))
            gm.tabprint("model: %s"%mName)
        return None
        
    elif printAll: print("MS: perturbation_compare: Read model image: %s"%mName)
        
        
    # GET INIT IMAGE    
    iImg = None
    iLoc = None
    
    # Create dict for storing init images if needed later    
    if rInfo.get('initImg') == None:
        rInfo.initImg = {}
        
    # Check if in rInfo has model Image already
    if type( rInfo.initImg.get(mName) ) != type(None):
        iImg = rInfo.initImg.get(mName)
    
    # Else, have rInfo retrieve img location
    else:
        iImg = rInfo.getModelImage(mName,initImg=True)
        rInfo.initImg[mName] = iImg
        
    if type(iImg) == type(None): 
        if printBase: 
            print("MS: Error: target_image_compare: failed to load init image")
            gm.tabprint("runId: %s"%rInfo.get('run_id'))
            gm.tabprint("model: %s"%mName)
        return None
        
    elif printAll: print("MS: perturbation_compare: Read init image: %s"%mName)
    
    # Create score and add to rInfo
    
    score = dc.createScore( mImg, iImg, param['cmpArg'] )
    newScore = rInfo.addScore( name = pName, score=score )
    
    return score


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
