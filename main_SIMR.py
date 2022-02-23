#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author:	 Matthew Ogden
    Created:	01 Sep 2020
Description:	Hopefully my primary code for calling all things SPAM Simulation, Modeling, and Analysis.
'''

# Python module imports
from os import path, listdir
from sys import path as sysPath, exit
sysPath.append( path.abspath( 'Machine_Score/' ) )

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
import Simulator.main_simulator as sm
import Image_Creator.main_image_creator as ic
import Feature_Extraction.main_feature_extraction as fe
import Neural_Network.main_neural_networks as nn
from Machine_Score import main_machine_score as ms
import Score_Analysis.main_score_analysis as sa


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
        if mpi_rank == 0 and arg.printBase:
            print( 'SIMR: main: In MPI environment!')
        sleep(1)
        if arg.printAll: gm.tabprint('I am %d of %d ' %( mpi_rank, mpi_size ) ) 
        mpi_comm.Barrier()
        
    if arg.printAll and mpi_rank == 0:
        arg.printArg()
        gm.test()
        sm.test()
        im.test()
        fe.test()
        nn.test()
        ms.test()

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
    
    # Normal local machine runs
    if mpi_size == 1:
        
        for folder in targetList:  
            
            tArg.targetDir = dataDir + folder            
            simr_target( arg = tArg )

    
    elif mpi_size > 1:

        for folder in targetList:  
            
            tInfo = None
            if mpi_rank == 0:
                tArg.targetDir = dataDir + folder
                tInfo = im.target_info_class( tArg=tArg )
            
                # If creating a new base, create new images
                if arg.get('newBase',False):
                    chime_0 = tInfo.readScoreParam( 'chime_0' )
                    chime_image = ic.adjustTargetImage( tInfo, chime_0['chime_0'] \
                                                       , printAll = arg.printAll )
                    tInfo.saveWndchrmImage( chime_image, chime_0['chime_0']['imgArg'] )
                
                if tInfo.status:                    
                    if printAll: gm.tabprint("Rank %d sending tInfo: %s" % (mpi_rank, tInfo.get('target_id')))
                    tInfo = mpi_comm.bcast( tInfo, root=0 )
                    
                else:
                    if printAll: gm.tabprint("Rank %d sending None:" % (mpi_rank))
                    tInfo = mpi_comm.bcast( None, root=0 )
                    
            else:
                tInfo = mpi_comm.bcast( tInfo, root=0 )

            if type( tInfo ) != type( None ):
                if printAll: gm.tabprint("Rank %d received: %s" % (mpi_rank, tInfo.get('target_id')))
                simr_target( tInfo = tInfo, arg = tArg )
            elif printAll:
                gm.tabprint("Rank %d received None:"%mpi_rank)

# End data dir


# Process target directory
def simr_target( arg=gm.inArgClass(), tInfo = None ):

    tDir = arg.targetDir
    printBase = arg.printBase
    printAll = arg.printAll
    
    if arg.printAll: 
        arg.printBase = True

    if printAll and mpi_rank == 0:
        print("SIMR: simr_target: input")
        print("\t - tDir: %s" % tDir )
        print("\t - tInfo: %s" % type(tInfo) )

    # Check if given a target
    if tInfo == None and tDir == None and arg.get('tInfo') == None:
        print("WARNING: SIMR: simr_target")
        print("\t - Please provide either target directory or target_info_class")
        return

    # Read target directory if location given. 
    elif tInfo == None and tDir != None:
        
        if mpi_rank == 0:
            
            tInfo = im.target_info_class( targetDir=tDir, tArg = arg )
            
            # If creating a new base, create new images
            if arg.get('newBase',False):
                chime_0 = tInfo.readScoreParam( 'chime_0' )
                chime_image = ic.adjustTargetImage( tInfo, chime_0['chime_0'], \
                                                   printAll = arg.printAll )
                tInfo.saveWndchrmImage( chime_image, chime_0['chime_0']['imgArg']['name'] )
                
            
            # Gather scores if called for
            if arg.get('update',False):
                tInfo.gatherRunInfos()
                tInfo.updateScores()
                tInfo.saveInfoFile()
            
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
        if tInfo.status: print("SIMR: target: %s - %s - %d Models" % ( tInfo.status, tInfo.get('target_id'), len( tInfo.tDict['zoo_merger_models'] ) ) )
        else: print("SIMR: target: %s - %s" % ( tInfo.status, tDir ) )

    # Check if valid directory
    if tInfo.status == False:
        print("SIMR: WARNING: simr_target:  Target Info status bad")
        return

    if arg.get('printParam', False) and mpi_rank == 0:
        tInfo.printParams()
    
    # Create new files/scores if called upon
    if arg.get( 'newSim', False ) or arg.get( 'newImage', False ) or arg.get( 'newFeats', False ) or arg.get( 'newScore', False ) or arg.get( 'normFeats', False ):
        new_target_scores( tInfo, arg )
        


def new_target_scores( tInfo, tArg ):

    printBase = tArg.printBase
    printAll = tArg.printAll
    
    if printBase and mpi_rank == 0:
        print("SIMR: new_target_scores: %s" % tInfo.get('target_id') )

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
        gm.tabprint( 'Rank %d in new_target_scores: %s: '% (mpi_rank, tInfo.get('target_id') ) )    
        mpi_comm.Barrier()
    
    runArgs = gm.inArgClass()
       
    if tArg.get('printAllRun',False):
        runArgs.setArg('printAll',True)
        
    elif tArg.get('printBaseRun',False):
        runArgs.setArg('printBase',True)
        
    else:
        runArgs.setArg('printBase',False)
    
    runArgs.setArg('tInfo', tInfo)
    runArgs.setArg('scoreParams', params)
    runArgs.setArg('newSim', tArg.get('newSim',False))
    runArgs.setArg('newImage', tArg.get('newImage',False))
    runArgs.setArg('newFeats', tArg.get('newFeats',False))
    runArgs.setArg('newNN', tArg.get('newNN',False))
    runArgs.setArg('newScore', tArg.get('newScore',False))
    runArgs.setArg('overWrite', tArg.get('overWrite',False))
    
    if printAll:
        print('SIMR.new_target_scores: Printing new run argments')
        runArgs.printArg()

    # Rank 0 has argList and will distribute
    argList = None    
    scatter_lists = []
    
    if mpi_rank == 0:
        
        # Find out which runs need new scores
        tInfo.gatherRunInfos()
        
        # Check if grabbing subset of runs.  If not, presume all
        if tArg.get('startRun',None) != None \
            and tArg.get('nRun',None) != None \
            and tArg.get('endRun',None) != None:
         
            runDicts = tInfo.iter_run_dicts()
        
        else:            
            runDicts = tInfo.iter_run_dicts( \
                                           startRun = tArg.get('startRun',0), \
                                           endRun = tArg.get('endRun',-1), \
                                           stepRun = tArg.get('stepRun',1),)
        
            
        argList = []
        for i,rKey in enumerate(runDicts):
            
            rScore = tInfo.get('zoo_merger_models')[rKey]['machine_scores']

            # Loop through wanted scores
            scoreGood = True
            for sKey in params:
                if rScore.get(sKey,None) == None:
                    scoreGood = False
                    break

            if not scoreGood or tArg.get('overWrite',False):
                rDir = tInfo.getRunDir(rID=rKey)
                if rDir == None:
                    if printBase: print("WARNING: Run invalid: %s" % rKey)
                argList.append( dict( arg = runArgs, rDir=rDir, ) )

        # Print how many scores expecting to be completed.
        if printBase:
            if len(argList) == 0:
                gm.tabprint("Scores already exist!")        
            else: 
                gm.tabprint("Runs needing scores: %d"%len(argList))

        # Divide up the big list into many lists for distributing
        if len( argList ) > 0:
            scatter_lists = [ argList[i::mpi_size] for i in range(mpi_size) ]

        # If nothing to do, create a list of empty lists so others know
        else: 
            for i in range(mpi_size):
                scatter_lists.append([])

        if printAll:
            print("SIMR: new_target_scores: MPI Scatter argList")
            gm.tabprint("Rank 0 argList: %d"%len(argList))
            gm.tabprint("Rank 0 scatter_lists: %d"%len(scatter_lists))
            for i,lst in enumerate(scatter_lists):
                gm.tabprint("Rank 0 list %d: %d"%(i,len(lst)))

    # Scatter argument lists to everyone
    argList = mpi_comm.scatter(scatter_lists,root=0)
    if printAll:
        gm.tabprint("Rank %d received: %d"%(mpi_rank,len(argList)))

    # Everyone go through their list and execute runs    
    for i,args in enumerate(argList):
        simr_run( **args )

        if mpi_rank == 0:
            gm.tabprint("Rank 0: Progress: %d / %d " % (i+1,len(argList)), end='\r')

    if mpi_rank == 0: print('')

    # Everyone wait for everyone else to finish
    mpi_comm.Barrier()

    # Have rank 0 collect files and update scores
    if mpi_rank == 0:

        # Check if target needs to create feature values
        if tArg.get('newFeats',False):
            fe.wndchrm_target_all( tArg, tInfo )
            fe.reorganize_wndchrm_target_data( tArg, tInfo )
        
        # Normalize feature values after created for runs
        if tArg.get('normFeats',False):
            fe.target_collect_wndchrm_all_raw( tArg, tInfo = tInfo )
            fe.target_wndchrm_create_norm_scaler( tArg, tInfo, )
            
        # Create new neural network model if called for
        if tArg.get('newNN',False):
            nn.target_new_neural_network( tArg, tInfo )

        tInfo.gatherRunInfos()
        tInfo.updateScores()

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
        print("SIMR.simr_run: Inputs")
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
    
    # Check if score parameters were given
    if arg.get('paramLoc') != None and arg.get('scoreParams') == None:
        sParams = gm.readJson( arg.paramLoc )
        if sParams == None:
            if arg.printBase: print("ERROR: SIMR: simr_run: Error reading param file: %s"%arg.paramLoc )
        else:
            arg.scoreParams = sParams
    
    # If invalid, complain
    if arg.get('scoreParams',None) == None:
        if printBase:
            print("SIMR: WARNING: simr_run: params not valid")
        return   
    
    # Check if new files should be created/altered
    newAll = arg.get('newAll')

    # Asking for new simulation data?
    if arg.get('newSim') or newAll:
        sm.main_sm_run( rInfo = rInfo, cmdArg=arg )

    # Asking for new image creation?
    if arg.get('newImage') or newAll:
        ic.main_ic_run( rInfo = rInfo, arg=arg )
        
    # Asking for new feature extraction from images? 
    if arg.get('newFeats') or newAll:
        fe.main_fe_run( rInfo = rInfo, arg=arg )

    # Asking for new score based on sim, image, and/or feature extractions?
    if arg.get('newScore') or newAll:

        ms.MS_Run( printBase = printBase, printAll = printAll, \
                rInfo = rInfo, params = arg.get('scoreParams'), \
                arg = arg )
    
    # Clean up temporary files from run directory.
    rInfo.delTmp()

# end processing run


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    
    main( arg )
