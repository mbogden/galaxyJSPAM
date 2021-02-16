#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
	Author:	 Matthew Ogden
	Created:	01 Sep 2020
Description:	Hopefully my primary code for calling all things Galaxy Simulation
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

sysPath.append( path.abspath( 'Machine_Score/' ) )
from Machine_Score import main_machine_score as ms

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

	if arg.simple:
		if arg.printBase: 
			print("SIMR: Simple!~")
			print("\t- Nothing else to see here")

	elif arg.runDir != None:
		simr_run( arg )

	elif arg.targetDir != None:
		simr_target( arg )

	elif arg.dataDir != None:
		procAllData( arg )

	else:
		print("SIMR: Nothing selected!")
		print("SIMR: Recommended options")
		print("\t - simple")
		print("\t - runDir /path/to/dir/")
		print("\t - targetDir /path/to/dir/")
		print("\t - dataDir /path/to/dir/")

# End main

def readParamFile( paramLoc ):

	# Read param file
	pClass = im.group_score_parameter_class( \
			pLoc = getattr( arg, 'paramLoc', None ), \
		)


def procAllData( arg ):

	dataDir   = arg.dataDir
	printBase = arg.printBase
	printAll  = arg.printAll 

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

	tArg = gm.inArgClass()

	# Find target directories
	for folder in dataList:
		tDir = dataDir + folder
		print(tDir)



	print("SIMR: Printing Results")
	for folder in dataList:
		tDir = dataDir + folder

		tInfo = im.target_info_class( targetDir = tDir, )
		c, tc = tInfo.getScoreCount( pClass.get('name',None ) )

		print( '%5d / %5d - %s ' % ( c, tc, tInfo.get( 'target_identifier', 'BLANK' ) ) )

# End data dir


# Process target directory
def simr_target( arg=gm.inArgClass(), tInfo = None ):

    tDir = arg.targetDir
    printBase = arg.printBase
    printAll = arg.printAll

    if printBase:
        print("SIMR: pipelineTarget: input")
        print("\t - tDir: %s" % tDir )
        print("\t - tInfo: %s" % type(tInfo) )

    # Check if given a target
    if tInfo == None and tDir == None:
        print("SIMR: WARNING: pipelineTarget")
        print("\t - Please provide either target directory or target_info_class")
        return

    # Read target directory if location given. 
    elif tInfo == None:
        tInfo = im.target_info_class( targetDir=tDir, tArg = arg )

    if printBase:
        print("SIMR: pipelineTarget status:")
        print("\t - tInfo.status: %s" % tInfo.status )

    # Check if valid directory
    if tInfo.status == False:
        print("SIMR: WARNING: pipelineTarget:  Target Info status bad")
        return

    if arg.get('printParam', False):
        tInfo.printParams()

    # Gather scores if called for
    if arg.get('update',False):
        tInfo.gatherRunInfos()
        tInfo.updateScores()
        tInfo.saveInfoFile()
        
    
    newSim = arg.get('newSim',False)
    newImg = arg.get('newImg',False)
    newScore = arg.get('newScore',False)
    newAll = arg.get('newAll',False)
    
    # Check if score parameter file is valid. 
    if newSim or newImg or newScore or newAll:
        
        paramLoc = arg.get('paramLoc')
        
        # Check if location is given
        if not gm.validPath(paramLoc):
            if printBase:
                print("SIMR: WARNING: new_target: param location not valid")
                gm.tabprint('paramLoc: %s'%paramLoc)
            return

        # Read parameter file
        else:
            pClass = im.group_score_parameter_class(paramLoc)
            if pClass.status:
                params = pClass.get('group',None)
                del pClass

        # Check for valid parameter file
        if params == None:
            if printBase:
                print("SIMR: WARNING: Target_New: Failed to load parameter class")
                gm.tabprint('paramLoc: %s',paramLoc)
            return
        
        # Save parameters in argument class
        arg.setArg('params',params)     

    # Create new files/scores if called upon
    if arg.get('newAll') or arg.get('newScore') :
        new_target_scores( tInfo, arg )
        tInfo.updateScores()

        
        
def new_target_scores( tInfo, tArg ):

    printBase = tArg.printBase
    printAll = tArg.printAll
    params = tArg.get("params",None)

    if printBase:
        print("SIMR: newTargetScores:")
        print("\t - tInfo: %s" % tInfo.status )
        
    if params == None:
        if printBase: print("SIMR: WARNING: newTargetScores:  Please give valid score parameters")
        return None
    
    
    runDicts = tInfo.getAllRunDicts()
    
    # Prep arguments
    runArgs = gm.inArgClass()
    runArgs.setArg('printBase', False)
    runArgs.setArg('printAll', printAll)
    runArgs.setArg('newScore', tArg.get('newScore',False))
    runArgs.setArg('tInfo', tInfo)
    runArgs.setArg('scoreParams', params)
    runArgs.setArg('overWrite', tArg.get('overWrite',False))
    
    
    # Find out which runs need new scores
    argList = []
    for i,rKey in enumerate(tInfo.get('zoo_merger_models')):
        rScore = tInfo.get('zoo_merger_models')[rKey]['machine_scores']
        
        # Loop through wanted scores
        scoreGood = True
        for sKey in params:
            if rScore.get(sKey,None) == None:
                scoreGood = False
                break
        
        if not scoreGood:
            rDir = tInfo.getRunDir(rID=rKey)
            argList.append( dict( arg = runArgs, rDir=rDir, ) )
    
    # If emtpy, new scores not needed
    if len(argList) == 0 and not tArg.get('overWrite',False):
        if printBase: im.tabprint("Scores already exist")
        return
    
    else:
        if printBase: im.tabprint("Runs needing scores: %d"%len(argList))

    # Prepare and run parallel class
    ppClass = gm.ppClass( tArg.nProc, printProg=True )
    ppClass.loadQueue( simr.simr_run, argList )
    ppClass.runCores()
    
    # Save results
    tInfo.updateScores()
    tInfo.saveInfoFile()

# End processing target dir


def simr_run( arg = None, rInfo = None, rDir = None ):

    # Initialize variables
    if arg == None:
        print("SIMR: WARNING: No arg. Exiting")

    if rDir == None:
        rDir = arg.runDir
    params = arg.scoreParams
    printAll = arg.printAll
    printBase = arg.printBase

    if printBase:
        print("SIMR.pipelineRun: Inputs")
        print("\t - rDir:", rDir)
        print("\t - rInfo:", type(rInfo) )

    # Initialize info file
    if rInfo == None:
        rInfo = im.run_info_class( runDir=rDir, rArg=arg )

    if printBase:
        print('SIMR.pipelineRun: ')
        print('\t - rInfo: ', (rInfo) )

    if rInfo.status == False:
        print("SIMR.pipelineRun: WARNING: runInfo bad")
        return

    if printBase:
        print("SIMR: run: scores before")
        rInfo.printScores()
    
    # Check if new files should be created/altered
    newSim = arg.get('newSim')
    newImg = arg.get('newImg')
    newScore = arg.get('newScore')
    newAll = arg.get('newAll')
    
    if newSim or newAll:
        if printBase: print("SIMR: run: newSim not functioning at this time")
        
    if newImg or newAll:
        if printBase: print("SIMR: run: newImg not functioning at this time")

    if newScore or newAll:
        
        ms.MS_Run( printBase = printBase, printAll = printAll, \
                rInfo = rInfo, params = arg.get('scoreParams'), \
                arg = arg )
    
    if arg.get('tInfo',None) != None:
        arg.tInfo.addRunDict(rInfo)

# end processing run



# Run main after declaring functions
if __name__ == '__main__':
	from sys import argv
	arg = gm.inArgClass( argv )
	main( arg )
