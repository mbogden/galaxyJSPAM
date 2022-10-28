'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    03 Sep 2020
Description:    For all things from a machine score out of the JSPAM simulation data.
'''

from os import path, listdir
from sys import exit, argv, path as sysPath

import numpy as np
import cv2
import pandas as pd
from copy import deepcopy

# For importing my useful support module
supportPath = path.abspath( path.join( __file__, "../../Support_Code/" ) )
sysPath.append( supportPath )
sysPath.append( __file__[:-21] )  # The 21 originates from the name of this file being 21 chars long "main_machine_score.py"

import general_module as gm
import info_module as im
import direct_image_compare as dc
import masked_image_compare as mc
import feature_image_compare as fc

def test():
    print("MS: Hi!  You're in Matthew's SIMR module for all things machine scoring images")

# global variables
test_func = None

# When creating and testing new functions from outside of module (Ex. Jupyter)
def set_test_func( inFuncLink ):
    global test_func
    test_func = inFuncLink

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
       
    overwrite = arg.get('overWrite',False)

    # Assume group of parameters and loop through.    
    for pKey in params:
        
        # Check if score exists
        score = rInfo.getScore( pKey )
        if printAll: print("MS: scoreName: %s"%pKey)
        
        # If score exists, move to next score parameter
        if score != None and not overwrite:
            continue
        
        # Grab parameter and score type
        param = params[pKey]
        
        # Check if score parameter if valid
        scoreType = param.get('scoreType',None)
        cmpArg = param.get('cmpArg',None)
        if scoreType == None or cmpArg == None:
            if printBase: 
                print("WARNING: MS: Score Parameters invalid: %s"%param.get('name',None))
            continue
        
        cmpType = param['cmpArg'].get('type',None)
        
        if cmpType == None:
            if printBase: 
                print("WARNING: MS: Score Parameters invalid: %s"%param.get('name',None))
            continue
        
        # Call function with correct score type
        if scoreType == 'model_fitness_score':
            
            if cmpType == 'direct_image_comparison':
                score_target_compare( rInfo, param, arg )
                
            elif cmpType == 'multi_image_compare':
                multi_image_comparison( rInfo, param, arg )
                
            elif cmpType == 'mask_binary_simple_compare':
                mask_compare_setup( rInfo, param, arg )
                
            elif cmpType == 'feature_compare':
                tImg, mImg = target_compare_setup( rInfo, param, arg )
                if type(tImg) == type(None) or type(mImg) == type(None):
                    continue
                    
                score = fc.create_feature_score( tImg, mImg, cmpArg )       
                
                # Add score if valid score
                if score != None:
                    newScore = rInfo.addScore( name = param['name'], score=score )    
                    if printAll: print("MS: mask_compare_setup: New Score!: %s - %f - %f" % (param['name'],score, newScore))
        
                else:
                    print("WARNING: MS: New Score is None: %s - %s " % (rInfo.get('run_id'),param['name']))

            else:
                if printBase:
                    print("WARNING: MS: MS_Run: Scoring method not implemented: %s"%cmpType)
                    
        elif scoreType == 'perturbation':
            perturbation_compare_setup( rInfo, param, arg )
            
        else:
            print("WARNING: MS: Run")
            print("Score Type not yet implemented: '%s'"%(scoreType))
            
    # Save results
    rInfo.saveInfoFile()
    
# end processing run dir
   
        

def multi_image_comparison( rInfo, param, arg ):
    
    printBase = arg.printBase
    printAll = arg.printAll
    
    if printBase:  print('MS.multi_image_comparison:')
    if printAll:   gm.tabprint("score_name: %s" % param['name'] )
    
    imgName = param['imgArg']['name']
    tgtName = param['cmpArg']['targetName']
    
    if printAll:  gm.tabprint("image_name: %s" % imgName )
    if printAll:  gm.tabprint("target_name: %s" % tgtName )
    
    mImg = rInfo.getModelImage( imgName = imgName, overWrite = True )
    uImg = rInfo.getModelImage( imgName = imgName, imgType = 'init', overWrite = True )
    tImg = rInfo.tInfo.getTargetImage( tName = tgtName )
    
    # Check all images are valid
    if type( mImg ) == type( None ) \
    or type( uImg ) == type( None ) \
    or type( tImg ) == type( None ):
        print("ERROR: MS.multi_image_comparison:")
        gm.tabprint("Invalid image")
        gm.tabprint("Target: %s" % type(tImg) )
        gm.tabprint("Model : %s" % type(mImg) )
        gm.tabprint("Init  : %s" % type(uImg) )
        return
    
    # Reshape target image if needed. 
    if tImg.shape != mImg.shape:
        tImg = cv2.resize( tImg, ( mImg.shape[1], mImg.shape[0] ) )
        
    score = grahams_scoring_function( tImg, mImg, uImg, printAll = printAll )
    
    if printBase:
        gm.tabprint("New Score: %s - %s" % (param['name'], str( score ) ) )
    
    if score != None: 
        newScore = rInfo.addScore( name = param['name'], score=score )    
            
    else:
        print("WARNING: MS: New Score is None: %s - %s " % (rInfo.get('run_id'),param['name']))
    
    return score

# End multi_image_comparison

def covW( x, y, w ):
    n = len(x)
    return np.sum( w * (x - np.average(x,weights=w)) * (y - np.average(y,weights=w)) )/np.sum(w)*n/(n-1)
# end

def corrW( x, y, w ):
    return covW(x,y,w)/( covW(x,x,w)*covW(y,y,w) )**0.5
# end

def perturb( pm, pt ):
    if( pm <= pt ):
        r = pm/pt
    else:
        r = (1.0-pm)/(1.0-pt)
    # end
    return r
# end


def grahams_scoring_function( tImg, mImg, uImg, h = 0.0, bin_img = False, printAll = False ):
    
    if printAll:
        print("MS.grahams_scoring_function")
        gm.tabprint('tImg: %s' % str( tImg.shape ) )
        gm.tabprint('mImg: %s' % str( mImg.shape ) )
        gm.tabprint('uImg: %s' % str( uImg.shape ) )
     
    # Copy and preprocess images
    T = deepcopy( tImg )
    M = deepcopy( mImg )
    U = deepcopy( uImg )
    
    if bin_img:
        # Make pixel values 0 or 1. 
        T[T>h] = 1.0
        M[M>h] = 1.0
        U[U>h] = 1.0
    
    T = np.log( 1+T.flatten() )
    M = np.log( 1+M.flatten() )
    U = np.log( 1+U.flatten() )
    
    # Get weights by differences bewteen target, model, unperturbed model. 
    tm = np.abs(T-M)
    mu = np.abs(M-U)
    tu = np.abs(T-U)
    
    weights = ( tm + mu + tu ) + np.ones(len(tu))*np.mean(tm+mu+tu)
    
    if printAll:
        gm.tabprint("weights")
        print(weights)

    # F1
    tmScore = corrW( T, M, weights )
    if( tmScore < 0 ): tmScore = 0.0
    
    # F2
    muScore  = np.corrcoef( M, U )[0,1]
    if( muScore < 0.01 ):  muScore = 0.01
    
    tuScore  = np.corrcoef( T, U )[0,1]
    if( tuScore < 0.01 ):  tuScore = 0.01
    muScoreX = perturb( muScore, tuScore )
        
        
    # Final score
    score    = tmScore*muScoreX
    if( score < 0.01 ):  score = 0.01
    
    if printAll:
        gm.tabprint("tmScore : %s" % str( tmScore ) )
        gm.tabprint("muScore : %s" % str( muScore ) )
        gm.tabprint("tuScore : %s" % str( tuScore ) )
        gm.tabprint("muScoreX: %s" % str( muScoreX ) )
        gm.tabprint("score   : %s" % str( score ) )
    
    return score
# End grahams_scoring_function

def target_compare_setup( rInfo, param, args): 
    # Get variables
    printBase = args.printBase
    printAll = args.printAll
    
    # Base info
    pName = param['name']
    tName = param['cmpArg']['targetName']
    mName = param['imgArg']['name']
    
    if printAll: print('MS: target_compare_setup: %s'%pName)
    
    if printAll:
        gm.tabprint(' paramName: %s'%pName)
        gm.tabprint(' modelName: %s'%mName)
        gm.tabprint('targetName: %s'%tName)
    
    # Get Target info
    tInfo = rInfo.get('tInfo')
    
    # Function now requires a target info class
    if tInfo == type(None):
        if printBase:
                print("ERROR: MS: target_compare_setup: Invalid target")
        return None, None
        
    # GET TARGET IMAGE
    tImg = tInfo.getTargetImage(tName)
    
    # Finally, should have tImg.  Leave if not
    if type(tImg) == None:
        if printBase: print("Error: MS: target_compare_setup: failed to load target image")
        return None, None
    elif printAll: 
        gm.tabprint("Read target image")
    
    # GET MODEL IMAGE    
    mImg = rInfo.getModelImage( mName )
        
    if type(mImg) == type(None):
        if printBase: 
            print("Error: MS: target_compare_setup: failed to load model image")
            gm.tabprint("runID - model: %s - %s"% (rInfo.get('run_id'),mName))
        return None, None
        
    elif printAll: 
        gm.tabprint("Read model image")
    
    # Check if all images have the same size
    if not ( mImg.shape == tImg.shape ):
        if printBase: 
            print("WARNING: MS: mask_compare_setup: Image shapes are not the same:")
            gm.tabprint("mImg: %s"%str(mImg.shape))
            gm.tabprint("tImg: %s"%str(tImg.shape))
        return None, None
    
    # Everything should be good!
    return tImg, mImg
    

# compares a target and image with given score parameters
def score_target_compare( rInfo, param, args ):
    
    # Get variables
    printBase = args.printBase
    printAll = args.printAll
    
    # Base info
    pName = param['name']
    tName = param['cmpArg']['targetName']
    mName = param['imgArg']['name']
    
    if printAll: print('MS: target_compare_setup: %s'%pName)
    
    if printAll:
        gm.tabprint(' paramName: %s'%pName)
        gm.tabprint(' modelName: %s'%mName)
        gm.tabprint('targetName: %s'%tName)
    
    # Get Target info
    tInfo = rInfo.get('tInfo')
    
    # Function now requires a target info class
    if tInfo == type(None):
        if printBase:
                print("ERROR: MS: target_compare_setup: Invalid target")
        return None
        
    # GET TARGET IMAGE
    tImg = tInfo.getTargetImage(tName)
    
    # Finally, should have tImg.  Leave if not
    if type(tImg) == type(None):
        if printBase: print("Error: MS: target_compare_setup: failed to load target image")
        return None
    elif printAll: 
        gm.tabprint("Read target image")
    
    # GET MODEL IMAGE    
    mImg = rInfo.getModelImage( mName )
        
    if type(mImg) == type(None): 
        if printBase: 
            print("Error: MS: target_compare_setup: failed to load model image")
            gm.tabprint("runID - model: %s - %s"% (rInfo.get('run_id'),mName))
        return None
        
    elif printAll: 
        gm.tabprint("Read model image")
    
    # Check if all images have the same size
    if not ( mImg.shape == tImg.shape ):
        if printBase: 
            print("WARNING: MS: mask_compare_setup: Image shapes are not the same:")
            gm.tabprint("mImg: %s"%str(mImg.shape))
            gm.tabprint("tImg: %s"%str(tImg.shape))
        return None    
    
    
    # Create score and add to rInfo
    score = dc.createScore( tImg, mImg, param['cmpArg'], printBase=rInfo.printBase )
    
    if score != None: 
        newScore = rInfo.addScore( name = param['name'], score=score )    
        if printAll: print("MS: New Score!: %s - %f - %f" % (param['name'],score, newScore))
            
    else:
        print("WARNING: MS: New Score is None: %s - %s " % (rInfo.get('run_id'),param['name']))
    
    return score

# compares a target and image with given score parameters
def mask_compare_setup( rInfo, param, args ):
    
    # Get variables
    printBase = args.printBase
    printAll = args.printAll
    
    # Base info
    pName = param['name']
    tName = param['cmpArg']['targetName']
    mName = param['imgArg']['name']
    maskName = param['cmpArg']['mask']['name']
    maskType = param['cmpArg']['mask']['type']
    
    if printAll: print('MS: mask_compare_setup: %s'%pName)
    
    if printAll:
        gm.tabprint(' paramName: %s'%pName)
        gm.tabprint(' modelName: %s'%mName)
        gm.tabprint('targetName: %s'%tName)
        gm.tabprint('  maskName: %s'%maskName)
    
    # Get Target info, Function now requires a target info class
    tInfo = rInfo.get('tInfo')    
    if tInfo == type(None):
        if printBase:
                print("ERROR: MS: mask_compare_setup: Requires a target info class")
        return None
        
    # GET TARGET IMAGE, leave if unable to get
    tImg = tInfo.getTargetImage(tName)
    
    if type(tImg) == type(None):
        if printBase: print("Error: MS: mask_compare_setup: failed to load target image")
        return None
    elif printAll: 
        gm.tabprint("Read target image")
    
    # GET MODEL IMAGE    
    mImg = rInfo.getModelImage( mName )
        
    if type(mImg) == type(None): 
        if printBase: 
            print("Error: MS: mask_compare_setup: failed to load model image")
            gm.tabprint("runID - model: %s - %s"% (rInfo.get('run_id'),mName))
        return None        
    elif printAll: 
        gm.tabprint("Read model image")
    
    if maskType == 'target':
        # Get target mask
        mask = tInfo.getMaskImage(maskName)

        if type(mask) == type(None):
            if printBase: print("Error: MS: mask_compare_setup: failed to load target mask")
            return None
        elif printAll: 
            gm.tabprint("Read target mask")
    else:
        if printBase: print("WARNING: MS: mask_compare_setup: cannot load mask type: %s"%maskType)
        return None
    
    
    # Check if all images have the same size
    if not ( mImg.shape == tImg.shape and tImg.shape == mask.shape ):
        if printBase: 
            print("WARNING: MS: mask_compare_setup: Image shapes are not the same:")
            gm.tabprint("mImg: %s"%str(mImg.shape))
            gm.tabprint("tImg: %s"%str(tImg.shape))
            gm.tabprint("mask: %s"%str(mask.shape))
        return None
    
    # Create score and add to rInfo
    
    score = mc.mask_binary_simple_compare( tImg, mImg, mask, param['cmpArg'] )
    
    if score != None:
        newScore = rInfo.addScore( name = param['name'], score=score )    
        if printAll: print("MS: mask_compare_setup: New Score!: %s - %f - %f" % (param['name'],score, newScore))
        
    else:
        print("WARNING: MS: New Score is None: %s - %s " % (rInfo.get('run_id'),param['name']))
    
    return score


def perturbation_compare_setup( rInfo, param, args ):
    
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
        iImg = rInfo.getModelImage(mName, imgType='init')
        rInfo.initImg[mName] = iImg
        
    if type(iImg) == type(None): 
        if printBase: 
            print("MS: Error: target_image_compare: failed to load init image")
            gm.tabprint("runId: %s"%rInfo.get('run_id'))
            gm.tabprint("model: %s"%mName)
        return None
        
    elif printAll: print("MS: perturbation_compare: Read init image: %s"%mName)
    
    # Create score and add to rInfo
    
    score = dc.createScore( mImg, iImg, param['cmpArg'], printBase=rInfo.printBase )
    
    if score != None:
        newScore = rInfo.addScore( name = param['name'], score=score )    
        if printAll: print("MS: mask_compare_setup: New Score!: %s - %f - %f" % (param['name'],score, newScore))
        
    else:
        print("WARNING: MS: New Score is None: %s - %s " % (rInfo.get('run_id'),param['name']))
    
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
