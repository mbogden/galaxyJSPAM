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
from copy import deepcopy


import pandas as pd 
import numpy as np 
import cv2 
from time import sleep 
from mpi4py import MPI 
from mpi_master_slave import Master, Slave, WorkQueue

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
sysPath.append( path.abspath( 'Machine_Score/' ) )
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
        target_main( arg )

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


# Process target directory
def target_main( cmdArg=gm.inArgClass(), tInfo = None ):
    
    printAll = cmdArg.printAll
    printBase = cmdArg.printBase
    
    if cmdArg.printAll:
        print("SIMR: target_main:")

    # Initialize target_info from disk and cmd arguments
    if mpi_rank == 0 and tInfo == None:        
        tInfo = target_initialize( cmdArg, tInfo )
        
    # If in mpi_env, send to others
    if mpi_size > 1:
        
        if mpi_rank == 0:
            if printBase: print("SIMR: target_main: sending tInfo to others")
            tInfo = mpi_comm.bcast( tInfo, root=0 )

        # others waiting for target info class. 
        else:
            tInfo = None
            tInfo = mpi_comm.bcast( tInfo, root=0 )

    # Have all check for valid tInfo
    if tInfo == None or tInfo.status == False:
        return None
    
    # If asking for genetic algorithm
    if cmdArg.get( 'gaExp', False ):
        target_geneteric_algorithm( tInfo, cmdArg )
    
    # Create new files/scores if called upon
    elif cmdArg.get( 'newAll', False ) \
    or cmdArg.get( 'newSim', False ) \
    or cmdArg.get( 'newImage', False ) \
    or cmdArg.get( 'newFeats', False ) \
    or cmdArg.get( 'newScore', False ) \
    or cmdArg.get( 'normFeats', False ):
        if printBase: 
            print("SIMR.simr_target:  Creating new scores")
            
        target_new_scores( tInfo, cmdArg )
    
    elif cmdArg.get( 'newGen', False ):
        
        target_new_gen_scores( cmdArg, tInfo )
    else:
        if printBase and mpi_rank == 0: 
            print("SIMR.simr_target:  Nothing Selected.")
     

class new_score_worker(Slave):

    def __init__(self):
        super(new_score_worker, self).__init__()
        print("Worker %d: __init__:" % mpi_rank )

    def do_work(self, data):
        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()
        run_id, mData = data
        print('Worker %d: do_work: received %d' % (mpi_rank, run_id), mData )
        sleep( 0.1 * run_id )
        return (True, 'I printed my task (%d)' % run_id)
            
def target_new_gen_scores( cmdArgs, tInfo ):
    
    
    printBase = tInfo.printBase
    printAll = tInfo.printAll
    
    if printBase:
        sleep(mpi_rank*0.25)
        print( "SIMR.target_new_gen_scores: %d of %d" %(mpi_rank,mpi_size))
    
    # Make sure you're in an MPI_environment with more than 1 core.
    if mpi_size == 1:
        print("WARNING: SIMR.target_new_gen_scores:")
        gm.tabprint("Please run code with mpirun to work properly")
        gm.tabprint("Example: 'mpirun -n 4 python3 main_simr.py -newGen'")
        return
    
    zScores, mData = (None, None)
    
    # If Master create queue and use workers
    if mpi_rank == 0:
        
        print("WORKING: SIMR.target_main")
        gm.tabprint("Using zoo merger data for testing")
        zScores, mData = tInfo.getOrbParam()
        newData = mData[0:15,:]
        
        master = Master( range(1, mpi_size ) )
         
        # Create Queue
        mpi_queue = WorkQueue( master )
        
        for i in range( newData.shape[0] ):
            mpi_queue.add_work( data=( i, newData[i,:] ) )
        
        # Stay in loop until queue is empty
        while not mpi_queue.done():
            
            # tell a slave to do work if needed.
            mpi_queue.do_work()
            for slave_return_data in mpi_queue.get_completed_work():
                done, message = slave_return_data
                if done:
                    #print('Master: slave finished is task and says "%s"' % message)
                    pass
            pass
            
        master.terminate_slaves()
    
    else:
        gm.tabprint(" %d: Creating new_score_worker" % mpi_rank)
        new_score_worker().run()
        
        
    

def target_initialize( cmdArg=gm.inArgClass(), tInfo = None ):
    
    
    tDir = cmdArg.targetDir
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
    
    if cmdArg.printAll: 
        cmdArg.printBase = True

    if printAll:
        print("SIMR.target_initialize:")
        print("\t - tDir: %s" % tDir )
        print("\t - tInfo: %s" % type(tInfo) )

    # Check if given a target
    if tInfo == None and tDir == None and cmdArg.get('tInfo') == None:
        print("WARNING: SIMR.simr_initialize_target:")
        print("\t - Please provide either target directory or target_info_class")
        return

    # Read target directory if location given. 
    elif tInfo == None and tDir != None:
        
        tInfo = im.target_info_class( targetDir=tDir, tArg = cmdArg )

        # If creating a new base, create new images
        if cmdArg.get('newBase',False):
            chime_0 = tInfo.readScoreParam( 'chime_0' )
            chime_image = ic.adjustTargetImage( tInfo, chime_0['chime_0'], \
                                               printAll = cmdArg.printAll )
            tInfo.saveWndchrmImage( chime_image, chime_0['chime_0']['imgArg']['name'] )


        # Gather scores if called for
        if cmdArg.get('update',False):
            tInfo.gatherRunInfoFiles()
            tInfo.updateScores()
            tInfo.saveInfoFile()

        
    # get target if in arguments. 
    elif tInfo == None and cmdArg.tInfo != None:
        tInfo = cmdArg.tInfo

    if printBase:
        if tInfo.status: 
            print("SIMR: target: %s - %s - %d Models" % ( tInfo.status, tInfo.get('target_id'), len( tInfo.tDict['zoo_merger_models'] ) ) )

            
        else: 
            print("SIMR: target: %s - %s" % ( tInfo.status, tDir ) )

    # Check if valid directory
    if tInfo.status == False:
        print("WARNING: SIMR.simr_target:  Target Info status bad")
        return None
            
    if cmdArg.get('printParam', False):
        tInfo.printParams()
        
    return tInfo
# End target_initialize
        

def target_new_scores( tInfo, tArg ):

    printBase = tArg.printBase
    printAll = tArg.printAll
    
    if printBase and mpi_rank == 0:
        print("SIMR: target_new_scores: %s" % tInfo.get('target_id') )

    # Check if parameter are given
    params = tArg.get('scoreParams')
    paramName = tArg.get('paramName',None)
    paramLoc = gm.validPath( tArg.get('paramLoc',None) )
    
    # If invalid, complain
    if params == None and paramLoc == None and paramName == None:
        if printBase and mpi_rank == 0:
            print("SIMR: WARNING: target_new_scores: params not valid")
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
            print("SIMR: WARNING: target_new_scores: Failed to load parameter class")
        return
    
    if tInfo.printAll and mpi_size > 1:
        gm.tabprint( 'Rank %d in target_new_scores: %s: '% (mpi_rank, tInfo.get('target_id') ) )    
        mpi_comm.Barrier()
    
    tArg.setArg('tInfo', tInfo)
    tArg.setArg('scoreParams', params)
    tArg.printArg()
    
    # Copy cmd Arg to send to runs
    runArgs = deepcopy( tArg )
    
    # Change printing variables
    if tArg.get('printAllRun',False):
        runArgs.setArg('printBase',True)
        runArgs.setArg('printAll',True)  
        
    elif tArg.get('printBaseRun',False):
        runArgs.setArg('printBase',True)
        runArgs.setArg('printAll',False)     
        
    else:
        runArgs.setArg('printBase',False)
        runArgs.setArg('printAll',False)
    
    if printAll:
        print('SIMR.target_new_scores: Printing new run argments\n')
        runArgs.printArg()
        print('')

    # Rank 0 has argList and will distribute
    argList = None    
    scatter_lists = []
    
    if mpi_rank == 0:
        
        # Find out which runs need new scores
        tInfo.gatherRunInfoFiles()
        
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
                argList.append( dict( cmdArg = runArgs, rDir=rDir, ) )

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
            print("SIMR: target_new_scores: MPI Scatter argList")
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

        tInfo.gatherRunInfoFiles()
        tInfo.updateScores()

# End processing target dir for new scores



def simr_run( cmdArg = None, rInfo = None, rDir = None ):

    # Initialize variables
    if cmdArg == None:
        print("SIMR: WARNING: No arg. Exiting")

    if rDir == None:
        rDir = cmdArg.runDir
        
    printAll = cmdArg.printAll
    printBase = cmdArg.printBase    
    
    if printBase:
        print("SIMR.simr_run: Inputs")
        print("\t - rDir:", rDir)
        print("\t - rInfo:", type(rInfo) )

    # Initialize info file
    if rInfo == None:
        
        if cmdArg.get('rInfo') != None:
            rInfo = cmdArg.rInfo
            
        elif rDir != None:
            rInfo = im.run_info_class( runDir=rDir, rArg=cmdArg )

    if printBase:
        print('SIMR.pipelineRun: ')
        print('\t - rInfo: ', (rInfo) )

    if rInfo.status == False:
        if printBase: print("SIMR.pipelineRun: WARNING: runInfo bad")
        return None
    
    # Check if score parameters were given
    if cmdArg.get('paramLoc') != None and cmdArg.get('scoreParams') == None:
        sParams = gm.readJson( cmdArg.paramLoc )
        if sParams == None:
            if cmdArg.printBase: print("ERROR: SIMR: simr_run: Error reading param file: %s"%cmdArg.paramLoc )
        else:
            cmdArg.scoreParams = sParams
    
    # If invalid, complain
    if cmdArg.get('scoreParams',None) == None:
        if printBase:
            print("SIMR: WARNING: simr_run: params not valid")
        return
    
    # Check for new scores to create
    overWrite =  cmdArg.get('overWrite', False) or cmdArg.get('overwrite', False)
    scoreParams = cmdArg.get('scoreParams',None)
    newParams = {}
    
    if overWrite:
        if rInfo.printAll: im.tabprint("Overwriting old scores")
        newParams = deepcopy( scoreParams )
        
    # Loop through score parameters and grab unique simulation scenarios. 
    else:
        for key in scoreParams:
            score = rInfo.getScore( scoreParams[key]['name'] )
            if score == None:
                newParams[key] = scoreParams[key]
        # End simKey in simParams
    
    # No new scores
    if len( newParams ) == 0:
        
        if rInfo.printBase:
            im.tabprint("No new scores")
        return
    
    else:
        if rInfo.printBase:
            gm.tabprint("Creating new scores")
        
    
    tmpArg = deepcopy( cmdArg )
    tmpArg.scoreParams = newParams

    # Check if new files should be created/altered    
    
    newAll = cmdArg.get('newAll',False)
    # Asking for new simulation data?
    if cmdArg.get('newSim') or newAll:
        sm.main_sm_run( rInfo = rInfo, cmdArg=tmpArg )

    # Asking for new image creation?
    if cmdArg.get('newImage') or newAll:
        ic.main_ic_run( rInfo = rInfo, arg=tmpArg )
        ''
    # Asking for new feature extraction from images? 
    if cmdArg.get('newFeats') or newAll:
        fe.main_fe_run( rInfo = rInfo, arg=tmpArg )

    # Asking for new score based on sim, image, and/or feature extractions?
    if cmdArg.get('newScore') or newAll:
        ms.MS_Run( printBase = printBase, printAll = printAll, \
                rInfo = rInfo, params = tmpArg.get('scoreParams'), \
                arg = tmpArg )
    
    # Clean up temporary files from run directory.
    rInfo.delTmp()

# end processing run


def simr_many_target( arg ):
    

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

# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    
    main( arg )
