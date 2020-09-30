'''
    Author:     Matthew Ogden
    Created:    01 Sep 2020
Description:    Hopefully my primary code for calling all things Galaxy Simulation
'''

# Python module imports
from os import path, listdir
from sys import path as sysPath

import pandas as pd
import numpy as np
import cv2

# For loading in Matt's general purpose python libraries
import Support_Code.general_module as gm
import Support_Code.info_module as im
import Simulator.main_simulator as ss

sysPath.append( path.abspath( 'Machine_Compare/' ) )
import Machine_Compare.main_compare as mc


def main(arg):

    if arg.printBase:
        print("SIMR: Hi!  You're in Matthew's main program for all things galaxy collisions")

    if arg.printAll:
        arg.printArg()
        gm.test()
        im.test()
        ss.test()

    # end main print

    # Read param file
    pClass = im.score_parameter_class( \
            paramLoc = getattr( arg, 'paramLoc', None ), \
            printBase = arg.printBase, \
            printAll = arg.printAll, \
            new = getattr( arg, 'newParam', False ), \
        )

    if pClass.status == False:
        print("SIMR.main: Bad param file. Exiting....")
        return

    elif arg.printBase:
        print("SIMR: param.status: Good")

    if arg.simple:
        if arg.printBase: 
            print("SIMR: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None:
        pipelineRun( \
                rDir = arg.runDir, \
                pClass = pClass, \
                printAll=arg.printAll, 
                newInfo = arg.get('newInfo', False ), \
                newScore = arg.get('newScore', False ), \
                targetLoc = arg.get( 'targetLoc', None ), \
                )

    elif arg.targetDir != None:
        pipelineTarget( arg.targetDir, \
                pClass = pClass, \
                newScore = arg.get('newScore', False ), \
                newInfo = arg.get('newInfo', False ), \
                newRunInfos = arg.get('newRunInfos', False ), \
                printAll=arg.printAll, )

    elif arg.dataDir != None:
        procAllData( arg.dataDir, param, printAll=arg.printAll )

    else:
        print("SIMR: Nothing selected!")
        print("SIMR: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

# End main
    
# Process target directory
def pipelineTarget( \
        tDir = None, pClass = None, \
        printBase = True, printAll=False, \
        newInfo = False, newRunInfos=False, \
        newScore = False, \
        ):

    if printBase:
        print("SIMR: pipelineTarget:")
        print("\t - tDir: " , tDir)

    tInfo = im.target_info_class( targetDir=tDir, \
            printBase = printBase, printAll=printAll, \
            newInfo = newInfo, newRunInfos = newRunInfos )

    if printBase:
        print("SIMR: pipelineTarget status:")
        print("\t - tInfo.status: %s" % tInfo.status )
        print("\t - param.status: %s" % pClass.status )

    # Check if target is valid
    if tInfo.status == False or pClass.status == False:
        print("SIMR: WARNING: pipelineTarget:  Bad status")
        return

    scores = tInfo.getScores()
    sName = pClass.get('name')
    pScores = scores[pClass.get('name')].values
    gScores = pScores[ np.invert( pd.isna( pScores ) ) ]

    nonScores = scores[ pd.isna( pScores ) ]

    print("SIMR: Target: Results: ")
    print("\t - score: %s" % pClass.get('name') )
    print('\t - found: %d / %d' % (len( gScores), len(pScores)))
    print('\t - create: %d' % len( nonScores ) )
    print( nonScores )

    if newScore:
        newTargetScores( tInfo, pClass, nonScores )


def newTargetScores( tInfo, pClass, nonScores ):

    sName = pClass.get('name')
    print("Creating new score")

    # Get target image
    tName = pClass.pDict['tgtArg']
    tLoc = tInfo.getTargetImage( tName = tName )
    tImg = mc.getImg( tLoc )

    for i, row in enumerate( nonScores.iterrows() ):

        runId = nonScores.run_id.iloc[ i]

        rInfo = tInfo.getRunClass( runId )

        mc.pipelineRun( rInfo = rInfo, param = pClass.pDict, tImg=tImg, printBase = True )

        tInfo.updateRunInfo( runId )

# End processing target dir


def pipelineRun( rDir = None, rInfo = None, \
        pClass = None, \
        printBase=True, printAll=False, \
        newScore = False, newInfo = False, \
        targetLoc = None, tImg = None, \
        ):

    if printBase:
        print("SIMR.pipelineRun: Inputs")
        print("\t - rDir:", rDir)
        print("\t - rInfo:", type(rInfo) )
        print("\t - param:", pClass != None)

    if rInfo == None:
        rInfo = im.run_info_class( runDir=rDir, \
                printBase = printBase, printAll=printAll,\
                newInfo = newInfo )

    if printBase:
        print('SIMR.pipelineRun: ')
        print('\t - param.name: ', pClass.pDict['name'] )
        print('\t - rInfo.status: ', rInfo.status )

    if rInfo.status == False:
        print("SIMR.pipelineRun: WARNING: runInfo bad")
        return

    score = rInfo.getScore( pClass.pDict.get('name') )

    if score != None:
        if printBase:
            print("SIMR: Score found:",score)
        return score

    else:
        if printBase: 
            print("SIMR.ppRun: Score not found")

        if newScore == False:
                print("\t - Please type '-new' in cmd to create score")
                return

        else:

            if printBase: ("SIMR: pipelineRun: Score not found, creating...")

            ptsLoc = procRunSim( rInfo, pClass.pDict['simArg'], \
                    printBase = printBase, printAll = printAll )

            if ptsLoc == None:
                print("SIMR: WARNING: pipelineRun: ")
                print("\t - New particle file not implemented")
                return 

            imgLoc = procRunImg( rInfo, pClass.pDict['imgArg'], \
                    printBase = printBase, printAll = printAll )

            if imgLoc == None:
                print("SIMR: WARNING: pipelineRun: ")
                print("\t - New image file not implemented")
                return

            if targetLoc == None and type(tImg) == type(None):
                print("SIMR: WARNING: pipelineRun: No target image given")
                print("\t -targetLoc path/to/target.png")
                return None

            if targetLoc != None:
    
                mc.pipelineRun( rInfo = rInfo, param = pClass.pDict, \
                        tLoc = targetLoc, imgLoc = imgLoc, printBase = printBase )

            elif type(tImg) != type(None):

                mc.pipelineRun( rInfo = rInfo, param = pClass.pDict, \
                        tImg = tImg, imgLoc = imgLoc, printBase = printBase )

            newScore = rInfo.getScore( pClass.pDict.get('name') )

            return newScore

# end processing run


# end procRunMach
def procRunMach( rInfo, srcArg, printBase = True, printAll = False, createPlots=True ):

    # Get desired number of particles for simulation
    sName = srcArg.get( 'name', None )
    print(srcArg)

    if sName == None:
        return None
    
    score = rInfo.getScore( sName, )


def procRunImg( rInfo, imgArg, printBase = True, printAll = False ):

    # Get desired number of particles for simulation
    imgParam = imgArg.get( 'name', None )

    if imgParam == None:
        print("SIMR.procRunImg: WARNING:")
        print("\t - Image name not found in parameter file")
        return None

    imgLoc = rInfo.findImgFile( imgParam )
    imgLoc, initLoc = rInfo.findImgFile( imgParam, initImg = True )

    if printBase:
        print('\t - Image name: ' , imgParam)
        print('\t - imgLoc: %s' % imgLoc)
        print('\t - initLoc: %s' % initLoc)

    if imgLoc == None:
        print("SIMR.procRunImg: WARNING:")
        print("\t - image not found")
        print("\t - creating new image not yet implemented")
        return None

    return imgLoc

def procRunSim( rInfo, simArg, printBase = True, printAll = False ):


    # Get desired number of particles for simulation
    dPts = simArg.get( 'name', None )

    if dPts == None:
        print("SIMR.procRunSim: WARNING:")
        print("\t - Particle file name not found in parameter file")
        return None

    ptsLoc = rInfo.findPtsFile( dPts )

    if printBase:
        print('\t - dPts: ' , dPts)
        print('\t - ptsLoc: %s' % ptsLoc)

    if ptsLoc == None:
        print("SIMR.procRunSim: WARNING:")
        print("\t - nPts file not found")
        print("\t - creating new file not yet implemented")
        return None

    return ptsLoc


# end processing run dir

def procAllData( dataDir, printBase=True, printAll=False ):

    from os import listdir

    if printBase: 
        print("SIMR.procAllData")
        print("\t - dataDir: %s" % arg.dataDir )

    # Check if directory exists
    if not path.exists( dataDir ):  
        print("SIMR.procAllData: WARNING: Directory not found")
        print('\t - ' , dataDir )
        return

    # Append trailing '/' if needed
    if dataDir[-1] != '/': dataDir += '/'  

    dataList = listdir( dataDir )   # List of items found in folder
    tDirList = []  # List of folders that are target directories

    # Find target directories
    for folder in dataList:
        tDir = dataDir + folder
        tempInfo = im.target_info_class( targetDir=tDir, printAll=False )

        # if a valid target directory
        if tempInfo.status:  tDirList.append( tempInfo )

    if printBase:
        print( '\t - Target Directories: %d' % len( tDirList ) )


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
