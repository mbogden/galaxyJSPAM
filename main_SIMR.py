#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def test():
    print("SIMR: Hi!  You're in Matthew's main program for all things galaxy collisions")

def main(arg):

    if arg.printBase:
        test()

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

    if arg.simple:
        if arg.printBase: 
            print("SIMR: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None:
        pipelineRun( pClass = pClass, arg = arg )

    elif arg.targetDir != None:

        pipelineTarget( pClass=pClass, arg=arg )

    elif arg.dataDir != None:
        procAllData( arg.dataDir, pClass=pClass, arg=arg )

    else:
        print("SIMR: Nothing selected!")
        print("SIMR: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

# End main
    

def procAllData( dataDir, pClass=None, arg = gm.inArgClass() ):

    printBase=arg.printBase
    printAll=arg.printAll 

    if printBase: 
        print("SIMR.procAllData")
        print("\t - dataDir: %s" % arg.dataDir )
        print("\t - pClass: %s" % pClass.status )

    # Check if valid string
    if type( dataDir ) != type( 'string' ):
        print("SIMR.procAllData: WARNING:  dataDir not a string")
        print('\t - %s - %s' %(type(dataDir), dataDir ) )
        return

    # Check if directory exists
    if not path.exists( dataDir ):  
        print("SIMR.procAllData: WARNING: Directory not found")
        print('\t - ' , dataDir )
        return

    dataDir = path.abspath( dataDir )

    # Append trailing '/' if needed
    if dataDir[-1] != '/': dataDir += '/'  
    dataList = listdir( dataDir )   # List of items found in folder

    tArgList = []

    if int( arg.get('nProc') ) == 1:

        # Find target directories
        for folder in dataList:
            tDir = dataDir + folder

            pipelineTarget( tDir = tDir, pClass = pClass, \
                    printBase = False, \
                    printAll = printAll, \
                    newInfo = arg.get( 'newInfo', False ), \
                    newRunInfos = arg.get('newRunInfos',False), \
                    newScore = arg.get('newScore', False), \
                    )

    # Prepare parellal processing
    else:

        # Find target directories
        for folder in dataList:
            tDir = dataDir + folder

            tArg = dict( tDir = tDir, pClass = pClass, \
                    printBase = False, \
                    printAll = printAll, \
                    newInfo = arg.get( 'newInfo', False ), \
                    newRunInfos = arg.get('newRunInfos',False), \
                    newScore = arg.get('newScore', False), \
                    )

            tArgList.append( tArg )

        # Initiate Pallel Processing
        nProcs = int( arg.get( 'nProc', 1 ) )
        print("SIMR: Requested %d cores" % nProcs)

        mp = gm.ppClass( nProcs )
        mp.printProgBar()
        mp.loadQueue( pipelineTarget, tArgList )
        mp.runCores()

    print("SIMR: Printing Results")
    for folder in dataList:
        tDir = dataDir + folder

        tInfo = im.target_info_class( targetDir = tDir, )
        c, tc = tInfo.getScoreCount( pClass.get('name',None ) )

        print( '%5d / %5d - %s ' % ( c, tc, tInfo.get( 'target_identifier', 'BLANK' ) ) )

# End data dir


# Process target directory
def pipelineTarget( pClass=None, arg=gm.inArgClass(), tInfo = None ):

    tDir = arg.targetDir
    printBase = arg.printBase
    printAll = arg.printAll
    newScore = arg.get('newScore',False)

    if printBase:
        print("SIMR: pipelineTarget: input")
        print("\t - tDir: %s" % tDir )
        print("\t - tInfo: %s" % type(tInfo) )
        print("\t - pClass: %s" % type(pClass) )

    if tInfo == None and tDir == None:
        print("SIMR: WARNING: pipelineTarget")
        print("\t - Please provide either target directory or target_info_class")
        return

    elif tInfo == None:

        tInfo = im.target_info_class( targetDir=tDir, \
                printBase = printBase, printAll=printAll, \
                newInfo = arg.get('newInfo',False), \
                newRunInfos = arg.get('newRunInfos',False), \
                )

    if printBase:
        print("SIMR: pipelineTarget status:")
        print("\t - tInfo.status: %s" % tInfo.status )

    if tInfo.status == False:
        print("SIMR: WARNING: pipelineTarget:  Target Info status bad")
        return

    if printBase:
        print("SIMR: Target prog before")
        #tInfo.printProg()

    # Get scores
    scores = tInfo.getScores()
    if type(scores) == type(None):
        print("SIMR: WARNING: No scores found")
        print("\t - Gathering runs")
        tInfo.gatherRunInfos()
        if type(tInfo.getScores) == None:
            print("SIMR: Score Error")
            return
    
    if newImage:
        print("HI!")
    
    # Create new scores if called upon
    if newScore:
        newTargetScores( tInfo, pClass )
        print("SIMR: Target progress after")
        tInfo.printProg()


def newTargetScores( tInfo, pClass, printBase = True, printAll = False, nonScores = None ):

    if printBase:
        print("SIMR: newTargetScores:")
        print("\t - tInfo: %s" % tInfo.status )
        print("\t - pClass: %s" % pClass.status )

    if pClass.status == False:
        if printBase: print("SIMR: Please provice valid pClass to create new scores")
        return
    
    sName = pClass.get('name')

    # Check if new scores are needed
    pDict = tInfo.tDict['progress']
    nRuns = pDict['zoo_merger_models']

    # Start fresh or appending?
    count, total = tInfo.getScoreCount( scrName = sName )
    if count == 0:
        tInfo.addScoreParam( pClass )
    
    # Initialize score creation 
    cmpType = pClass.get('cmpType',None)

    # Get target image
    if cmpType == 'target':
        tLoc = tInfo.findTargetImage( tName = pClass.get('targetName', None) )
        tImg = mc.getImg( tLoc )
        if type(tImg) == type(None):
            print("SIMR: newTargetScores: ERROR:")
            print("\t - Failed to read target img")
            print("\t - targetName: %s" % tName )

    elif cmpType == 'perturbation':
        tImg = None
        tLoc = None

    scores = tInfo.getScores()

    for i, row in scores.iterrows():

        rID = row['run_id']
        score = row[sName]

        if pd.isnull( row[sName] ):

            rInfo = tInfo.getRunInfo( rID = rID, printBase = printAll )
            score = mc.pipelineRun( rInfo = rInfo, pClass = pClass, \
                    tImg = tImg, printBase = printAll, )

            tInfo.addScore( rID, sName, score )

        if printBase:
            print(" New Scores: %d / %d" % ( i, nRuns ), end='\r' )

    tInfo.updateScoreProg()
    tInfo.saveInfoFile()


# End processing target dir


def pipelineRun( pClass = None, arg = None, rInfo = None, tImg = None, tInfoPtr=None ):

    # Initialize variables
    if arg == None:
        print("SIMR: WARNING: No arg. Exiting")

    rDir = arg.runDir
    printAll = arg.printAll 
    printBase = arg.printBase

    newInfo = arg.get('newInfo', False )
    newScore = arg.get('newScore', False )

    if printBase:
        print("SIMR.pipelineRun: Inputs")
        print("\t - rDir:", rDir)
        print("\t - rInfo:", type(rInfo) )
        print("\t - param:", pClass != None)

    # Initialize info file
    if rInfo == None:
        rInfo = im.run_info_class( runDir=rDir, \
                printBase = printBase, printAll=printAll,\
                newInfo = newInfo, tInfoPtr=tInfoPtr )

    if printBase:
        print('SIMR.pipelineRun: ')
        print('\t - param: ', type(pClass) )
        print('\t - rInfo: ', (rInfo) )

    if rInfo.status == False:
        print("SIMR.pipelineRun: WARNING: runInfo bad")
        return

    if printBase:
        print("SIMR: run: scores before")
        rInfo.printScores()

    if newScore:

        createNewRunScore( rInfo, pClass, \
                tImg = tImg, targetLoc = arg.get('targetLoc',None) )

        if printBase:

            print("SIMR: run: scores after")
            rInfo.printScores()

    # Should not reach
    return None


def createNewRunScore( rInfo, pClass, \
        printBase = True, printAll = False, \
        tImg = None, targetLoc = None ):

    # Create new score if variable is given
    if printBase: ("SIMR: createNewRunScore: Score not found, creating...")

    # Work backwords, get images if possible

    mcVal = mc.pipelineRun( printBase = printBase, printAll = printAll, \
            rInfo = rInfo, pClass = pClass, \
            tLoc = targetLoc, tImg = tImg )
    
# end processing run


def procRunImg( rInfo, imgArg, printBase = True, printAll = False ):

    # Get desired number of particles for simulation
    imgParam = imgArg.get( 'name', None )

    if imgParam == None:
        print("SIMR.procRunImg: WARNING:")
        print("\t - Image name not found in parameter file")
        return None


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

# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
