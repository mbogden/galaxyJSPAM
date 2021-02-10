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
		pipelineRun( arg )

	elif arg.targetDir != None:
		pipelineTarget( arg )

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
	paramLoc  = arg.get('paramLoc')

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
def pipelineTarget( arg=gm.inArgClass(), tInfo = None ):

	tDir = arg.targetDir
	printBase = arg.printBase
	printAll = arg.printAll

	newScore = arg.get('newScore',False)
	newImage = arg.get('newImage',False)
	newAll = arg.get('newAll', False)


	if printBase:
		print("SIMR: pipelineTarget: input")
		print("\t - tDir: %s" % tDir )
		print("\t - tInfo: %s" % type(tInfo) )

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
		im.tabprint( 'progress: ' + str( tInfo.get("progress") ) )

	if tInfo.status == False:
		print("SIMR: WARNING: pipelineTarget:  Target Info status bad")
		return

	if arg.get('printParam', False):
		tInfo.printParams()
		
	# Gather scores if called for
	if arg.get('updateScores',False):
		tInfo.updateScores()
	
	if newImage or newAll:
		print("HI!")
	
	# Create new scores if called upon
	if newScore or newAll:
		paramLoc = arg.get('paramLoc',None)
		paramClass = arg.get('paramClass',None)
		newTargetScores( tInfo, \
				printBase = printBase, printAll = printAll, \
				paramClass=paramClass, paramLoc = paramLoc )


def newTargetScores( tInfo, printBase = True, printAll = False,\
		pClass = None, paramLoc = None ):

	if printBase:
		print("SIMR: newTargetScores:")
		print("\t - tInfo: %s" % tInfo.status )
		im.tabprint('paramLoc: %s'%type(paramLoc))
		im.tabprint('paramClass: %s'%type(paramClass))

	if paramLoc == None and pClass == None:
		print("SIMR: WARNING: No score parameter class given")
		return

	elif pClass == None and paramLoc != None:
		pClass = readParamFile( paramLoc )
	
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


def simr_run( arg = None, rInfo = None ):

    # Initialize variables
    if arg == None:
        print("SIMR: WARNING: No arg. Exiting")

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
        rInfo = im.run_info_class( runDir=rDir, \
                printBase = printBase, printAll=printAll,\
                newInfo = newInfo, tInfoPtr=tInfoPtr )

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

# end processing run



# Run main after declaring functions
if __name__ == '__main__':
	from sys import argv
	arg = gm.inArgClass( argv )
	main( arg )
