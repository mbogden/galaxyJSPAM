#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author:	 Matthew Ogden
    Created:	01 Sep 2020
Description:	Hopefully my primary code for calling all things Galaxy Simulation
'''

# Python module imports
from os import path, listdir
from sys import path as sysPath, exit

import pandas as pd
import numpy as np
import cv2
from time import sleep
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD    
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# For loading in Matt's general purpose python libraries
import Support_Code.general_module as gm
import Support_Code.info_module as im
import Simulator.main_simulator as ss
import Score_Analysis.main_score_analysis as sa
import Image_Creator.main_image_creator as ic

sysPath.append( path.abspath( 'Machine_Score/' ) )
from Machine_Score import main_machine_score as ms

def test():
    print("SIMR: Hi!  You're in Matthew's main program for all things galaxy collisions")

# For testing and developement
new_func = None
def set_new_func( inFunc ):
    global new_func
    new_func = inFunc
# End set new function

def main(arg):

    if arg.printBase and mpi_rank == 0:
        test()

    if mpi_size > 1:     
        if mpi_rank == 0:
            print( 'SIMR: main: In MPI environment!')
        mpi_comm.Barrier()
        gm.tabprint('I am %d of %d ' %( mpi_rank, mpi_size ) )  
        sleep( mpi_rank * 0.1 )
        mpi_comm.Barrier()
        
    if arg.printAll and mpi_rank == 0:
        arg.printArg()
        gm.test()
        im.test()
        ss.test()

    # end main print

    if arg.simple:
        if arg.printBase and mpi_rank == 0: 
            print("SIMR: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None and mpi_rank == 0:
        simr_run( arg )

    elif arg.targetDir != None:
        simr_target( arg )

    elif arg.dataDir != None:
        simr_many_target( arg )

    elif mpi_rank == 0:
        print("SIMR: main: Nothing selected!")
        print("SIMR: main: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

# End main

def simr_many_target( arg ):
    
    from os import listdir
    from copy import deepcopy

    dataDir   = arg.dataDir
    printBase = arg.printBase
    printAll  = arg.printAll 

    if printBase and mpi_rank == 0: 
        print("SIMR.procAllData")
        print("\t - dataDir: %s" % arg.dataDir )
    
    # Check if valid directory
    dataDir = gm.validPath(dataDir)
    
    if dataDir == None:

        if printBase and mpi_rank == 0: 
            print("SIMR: WARNING: simr_many_target: Invalid data directory")
            print("\t - dataDir: %s" % dataDir )
            
            
    # Prep arguments for targets
    tArg = deepcopy(arg)
    tArg.dataDir = None
    tArg.printBase = True
    
    # Get list of directories/files and go through
    targetList = listdir( dataDir )   # List of items found in folder
    targetList.sort()
    
    if mpi_rank == 0:
        
        for folder in targetList:        

            tArg.targetDir = dataDir + folder
            tInfo = im.target_info_class( tArg=tArg )
            
            if tInfo.status:
                print("SIMR: many_targets: Good: %s" % tArg.get('target_id') )
                
                # Check if others are expecting this tInfo
                if mpi_size > 1:
                    tInfo = mpi_comm.bcast( tInfo, root=0 )
                    mpi_comm.Barrier()
                    print("Rank %d msg: %s" % (mpi_rank, tInfo.get('target_id')))
                    
                simr_target( arg = tArg, tInfo = tInfo )

            else:
                print("SIMR: many_targets: Bad Dir : %s" % tArg.targetDir )
            
            break
    else:
        msg = None
        tInfo = mpi_comm.bcast( msg, root=0 )
        mpi_comm.Barrier()
        print("Rank %d msg: %s" % (mpi_rank, tInfo.get('target_id')))
        simr_target( arg = tArg, tInfo = tInfo )
            
# End data dir


# Process target directory
def simr_target( arg=gm.inArgClass(), tInfo = None ):

    tDir = arg.targetDir
    printBase = arg.printBase
    printAll = arg.printAll
    
    if arg.printAll:
        arg.printBase = True

    if printBase and mpi_rank == 0:
        print("SIMR: pipelineTarget: input")
        print("\t - tDir: %s" % tDir )
        print("\t - tInfo: %s" % type(tInfo) )

    # Check if given a target
    if tInfo == None and tDir == None and arg.get('tInfo') == None:
        print("SIMR: WARNING: simr_target")
        print("\t - Please provide either target directory or target_info_class")
        return

    # Read target directory if location given. 
    elif tInfo == None and tDir != None:
        
        if mpi_rank == 0:
            
            tInfo = im.target_info_class( targetDir=tDir, tArg = arg )
            
            # Check if others are expecting this tInfo
            if mpi_size > 1:
                tInfo = mpi_comm.bcast( tInfo, root=0 )

        else:
            tInfo = None
            tInfo = mpi_comm.bcast( tInfo, root=0 )
        
    # get target if in arguments. 
    elif tInfo == None and arg.tInfo != None:
        tInfo = arg.tInfo

    if printBase and mpi_rank == 0:
        print("SIMR: simr_target status:")
        print("\t - tInfo.status: %s" % tInfo.status )

    # Check if valid directory
    if tInfo.status == False:
        print("SIMR: WARNING: simr_target:  Target Info status bad")
        return

    if arg.get('printParam', False) and mpi_rank == 0:
        tInfo.printParams()

    # Gather scores if called for
    if arg.get('update',False) and mpi_rank == 0:
        tInfo.gatherRunInfos()
        tInfo.updateScores()
        tInfo.saveInfoFile()

    newImage = arg.get('newImage',False)
    newScore = arg.get('newScore') 
    newAll = arg.get('newAll',False)
    
    # Create new files/scores if called upon
    if newImage or newScore or newAll:
        new_target_scores( tInfo, arg )
    


def new_target_scores( tInfo, tArg ):

    printBase = tArg.printBase
    printAll = tArg.printAll
    
    if printBase and mpi_rank == 0:
        print("SIMR: new_target_scores:")
        print("\t - tInfo: %s" % tInfo.status )

    # Check if parameter are given
    params = tArg.get('scoreParams')
    paramName = tArg.get('paramName',None)
    paramLoc = gm.validPath( tArg.get('paramLoc',None) )
    
    # If invalid, complain
    if params == None and paramLoc == None and paramName == None:
        if printBase and mpi_rank == 0:
            print("SIMR: WARNING: new_target_scores: params not valid")
            gm.tabprint('Params: %s'%type(params))
            gm.tabprint('ParamName: %s'%paramName)
            gm.tabprint('ParamLoc : %s'%paramLoc)
        return
    
    # If params there, move on
    elif params != None:
        pass
    
    # If given a param name, assume target knows where it is.
    elif params == None and paramName != None:
        
        # If normal execution
        if mpi_size == 1:
            params = tInfo.readScoreParam(paramName)
        
        # If in mpi and rank 0
        elif mpi_rank == 0:
            params = tInfo.readScoreParam(paramName)
            mpi_comm.bcast( params, root=0 )

        # If in mpi and expecting info.
        else:
            params = None
            params = mpi_comm.bcast( params, root=0 )
    
    # If given param location, directly read file
    elif paramLoc != None:
        
        # If normal execution
        if mpi_size == 1:
            params = gm.readJson(paramLoc)
        
        # If in mpi and rank 0
        elif mpi_rank == 0:
            params = gm.readJson(paramLoc)
            mpi_comm.bcast( params, root=0 )

        # If in mpi and expecting info.
        else:
            params = None
            params = mpi_comm.bcast( params, root=0 )

            
    # Check for final parameter file is valid
    if params == None:
        if printBase:
            print("SIMR: WARNING: new_target_scores: Failed to load parameter class")
        return
    
    if tInfo.printAll and mpi_size > 1:  
        mpi_comm.Barrier()       
        sleep( mpi_rank*0.25 )
        print( 'Rank %d in target_mpi_runs: %s: '% (mpi_rank, tInfo.get('target_id')))    
        mpi_comm.Barrier()
    
    runArgs = gm.inArgClass()
    
    if printAll: 
        runArgs.setArg('printAll', True)
        runArgs.setArg('printBase', True)
    else:
        runArgs.setArg('printBase', False)
        
    
    runArgs.setArg('tInfo', tInfo)
    runArgs.setArg('scoreParams', params)
    runArgs.setArg('newImage', tArg.get('newImage',False))
    runArgs.setArg('newScore', tArg.get('newScore',False))
    runArgs.setArg('overWrite', tArg.get('overWrite',False))

    argList = None
    if mpi_rank == 0:
        
        # Find out which runs need new scores
        tInfo.gatherRunInfos()
        runDicts = tInfo.getAllRunDicts()
        argList = []
        for i,rKey in enumerate(tInfo.get('zoo_merger_models')):
            rScore = tInfo.get('zoo_merger_models')[rKey]['machine_scores']

            # Loop through wanted scores
            scoreGood = True
            for sKey in params:
                if rScore.get(sKey,None) == None:
                    scoreGood = False
                    break

            if not scoreGood or tArg.get('overWrite',False):
                rDir = tInfo.getRunDir(rID=rKey)
                argList.append( dict( arg = runArgs, rDir=rDir, ) )


    
    # If not in MPI environment, function normally.
    if mpi_size == 1:
        
        # If empty, new scores not needed
        if len(argList) == 0 and not tArg.get('overWrite',False):
            if printBase and mpi_rank == 0: im.tabprint("Scores already exist")
            return
        
        elif mpi_rank == 0:
            if printBase: im.tabprint("Runs needing scores: %d"%len(argList))
                
        # Prepare and run parallel class
        ppClass = gm.ppClass( tArg.nProc, printProg=True )
        ppClass.loadQueue( simr_run, argList )
        ppClass.runCores()

        # Save results
        tInfo.addScoreParameters( params, overWrite = tArg.get('overWrite',False) )
        tInfo.gatherRunInfos()
        tInfo.updateScores()
        tInfo.saveInfoFile()
        
    # If in MPI environment, distribute argument list evenly to others
    elif mpi_size > 1:
        
        # Rank 0 has argList and will distribute
        scatter_lists = []
        if mpi_rank == 0:
            
            if len( argList ) > 0:
                scatter_lists = [ argList[i::mpi_size] for i in range(mpi_size) ]
            
            else: 
                for i in range(mpi_size):
                    scatter_lists.append([])
            
            if printBase:
                print("SIMR: new_target_scores: MPI Scatter argList")
                gm.tabprint("Rank 0 argList: %d"%len(argList))
                gm.tabprint("Rank 0 scatter_lists: %d"%len(scatter_lists))
                for i,lst in enumerate(scatter_lists):
                    gm.tabprint("Rank 0 list %d: %d"%(i,len(lst)))
                
        # Scatter argument lists to everyone
        argList = mpi_comm.scatter(scatter_lists,root=0)
        if printBase:
            gm.tabprint("Rank %d received: %d"%(mpi_rank,len(argList)))
        
        # Everyone go through their list and execute runs    
        for i,args in enumerate(argList):
            simr_run( **args )
            
            if mpi_rank == 0:
                gm.tabprint("Rank 0: Progress: %d / %d " % (i,len(argList)), end='\r')
        
        # Everyone wait for everyone else to finish
        mpi_comm.Barrier()
        
        # Have rank 0 collect files and update scores
        if mpi_rank == 0:
            tInfo.gatherRunInfos()
            tInfo.updateScores()
            tInfo.saveInfoFile()

# End processing target dir for new scores



def simr_run( arg = None, rInfo = None, rDir = None ):

    # Initialize variables
    if arg == None:
        print("SIMR: WARNING: No arg. Exiting")

    if rDir == None:
        rDir = arg.runDir
        
    printAll = arg.printAll
    printBase = arg.printBase    
    
    if printBase:
        print("SIMR.pipelineRun: Inputs")
        print("\t - rDir:", rDir)
        print("\t - rInfo:", type(rInfo) )

    # Initialize info file
    if rInfo == None:
        
        if arg.get('rInfo') != None:
            rInfo = arg.rInfo
            
        elif rDir != None:
            rInfo = im.run_info_class( runDir=rDir, rArg=arg )

    if printBase:
        print('SIMR.pipelineRun: ')
        print('\t - rInfo: ', (rInfo) )

    if rInfo.status == False:
        if printBase: print("SIMR.pipelineRun: WARNING: runInfo bad")
        return None

    # Check if parameter are given
    scoreParams = arg.get('scoreParams',None)
    
    # If invalid, complain
    if scoreParams == None:
        if printBase:
            print("SIMR: WARNING: simr_run: params not valid")
        return   
    
    # Check if new files should be created/altered
    newSim = arg.get('newSim')
    newImage = arg.get('newImage')
    newScore = arg.get('newScore')
    newAll = arg.get('newAll')

    if newSim or newAll:
        if printBase: print("WARNING: SIMR: run: newSim not functioning at this time")

    if newImage or newAll:
        ic.main_ic_run( rInfo = rInfo, arg=arg )

    if newScore or newAll:

        ms.MS_Run( printBase = printBase, printAll = printAll, \
                rInfo = rInfo, params = arg.get('scoreParams'), \
                arg = arg )

    if arg.get('tInfo',None) != None:
        arg.tInfo.addRunDict(rInfo)
    
    # Clean up run directory if temporary files were created
    rInfo.delTmp()

# end processing run


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    
    main( arg )
