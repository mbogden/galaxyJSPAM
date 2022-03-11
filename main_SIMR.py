#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author:	 Matthew Ogden
    Created:	01 Sep 2020
Description:	Hopefully my primary code for calling all things SPAM Simulation, Modeling, and Analysis.
'''

# Python module imports
from os import path, listdir, mkdir
from sys import path as sysPath, exit
from copy import deepcopy
from shutil import rmtree


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
        sleep(mpi_rank*0.015)
        if mpi_rank == 0 and arg.printBase:
            print( 'SIMR: main: In MPI environment!')
        if arg.printBase: gm.tabprint('I am %d / %d on %s' %( mpi_rank, mpi_size, MPI.Get_processor_name() ) ) 
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
    
    # If performing new Genetic Algorithm Experiment
    if cmdArg.get( 'gaExp', False ):
        target_genetic_algorithm( cmdArg, tInfo )
    
    # For testing new score workers
    elif cmdArg.get( 'newGen', False ):
        target_test_new_gen_scores( cmdArg, tInfo )
    
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
    
    else:
        if printBase and mpi_rank == 0: 
            print("SIMR.simr_target:  Nothing Selected.")


class new_score_worker(Slave):
    
    rank = mpi_rank
    worker_dir = None

    def __init__(self, tInfo, runArgs ):
        
        super(new_score_worker, self).__init__()
        
        # Save info for creating scores later. 
        self.tInfo = tInfo
        self.runArgs = runArgs
        self.score_names = [ name for name in self.runArgs.scoreParams ]
        
        if self.runArgs.printBase:  
            print("Worker %d: __init__:" % mpi_rank )
        
        # Where to save new run data 
        if self.runArgs.get( 'gaLoc', 'target' ) == 'target':
            
            tmpDir = self.tInfo.get('tmpDir', None)
            if tmpDir == None:
                print("%d WARNING: new_score_worker.__init__: tmpDir not found")
            
            else:
                
                self.worker_dir = tmpDir + 'worker_dir_%s/' % str(self.rank).zfill(4)
                
                if gm.validPath( self.worker_dir ) == None:
                    print("%d creating dir: %s" % (self.rank, self.worker_dir))
                    mkdir( self.worker_dir )
            
                    
        # For Babbage, our local cluster
        elif self.runArgs.get( 'gaLoc', None ) == 'babbage':
            
            # Hardcoded location of the tmp dir for babbages local computers
            tmpBabbageDir = '/state/partition1/'
            #tmpBabbageDir = 'tmpDir/'
            
            # If invalid location, complain
            if gm.validPath( tmpBabbageDir ) == None:
                self.worker_dir = None
                print("WARNING %d: Invalid dir: %s" % ( mpi_rank, tmpBabbageDir ) )
            
            # Working local diskspace is valid
            else:
                
                # Hardcoded
                tmpDir = tmpBabbageDir + 'mbo2d_simr/'
                
                if gm.validPath( tmpDir ) == None:
                    
                    # Wait to create, I'm hoping to avoid multiple people creating it at the same time.
                    waitTime = np.random.uniform() * 10
                    sleep(waitTime)
                    
                    # Hoping only one person will create and everyone else 
                    if gm.validPath( tmpDir ) == None:
                        mkdir( tmpDir )
                        print("Worker %d created tmp dir: %s" % (mpi_rank, tmpDir ) )
                    
                # Everyone meet up again
                if gm.validPath( tmpDir ) != None:
                    
                    self.worker_dir = tmpDir + 'worker_dir_%s/' % str(self.rank).zfill(4)
                    print("%d creating dir: %s" % (self.rank, self.worker_dir))
                    mkdir( self.worker_dir )
                
                # Check if valid, if not create.
                    
        # Final command to ensure it's either a valid location or None.
        self.worker_dir = gm.validPath( self.worker_dir )

        
    def do_work(self, data):
        
        i, mData = data
        run_id = 'run_%s' % str(i).zfill(4)
        
        if self.runArgs.printAll:
            print('Worker %d: do_work: received %s' % (mpi_rank, run_id), )
            
        # Double check if Worker has a valid working directory
        if type( self.worker_dir ) == type( None ):
            return (i, None)
        
        # Create run variables
        runDir = self.worker_dir + '%s/' % run_id
        infoLoc = runDir + 'base_info.json'
        
        # Create string out of parameters
        model_string = ','.join( map(str, mData) )

        # Create the information file for the run/model
        mInfo = {}
        mInfo['run_id'] = run_id
        mInfo['model_data'] = model_string
        
        # Create model directories and files
        if gm.validPath( runDir ) == None:
            mkdir( runDir )
            
        gm.saveJson( mInfo, infoLoc )
        
        rInfo = im.run_info_class( runDir = runDir, rArg = self.runArgs )
        
        if rInfo.status == False:
            return (i, None)
        
        # Create simulation, image and compute score
        simr_run( cmdArg = self.runArgs, rInfo = rInfo )
        
        scores = []
        for name in self.score_names:
            scores.append( rInfo.getScore( name ) )
            
        # Remove directory to convserve disk space.
        rmtree( runDir )
        
        # Assuming only one score
        return (i, scores[0])
    
# End class new_score_worker

        
def target_genetic_algorithm( cmdArgs, tInfo ):
    
    ###################################
    #####   Worker Preperation   ######
    ###################################
    
    printBase = tInfo.printBase
    printAll = tInfo.printAll
    
    if printBase:
        print( "SIMR.target_genetic_algorithm: %d of %d" %(mpi_rank,mpi_size) )
      
    # Make sure you're in an MPI_environment with more than 1 core.
    if mpi_size == 1:
        print("WARNING: SIMR.target_genetic_algorithm:")
        gm.tabprint("Must run code in mpirun with more than 1 core to work properly")
        gm.tabprint("Example: 'mpirun -n 4 python3 main_simr.py -gaExp'")
        return
    
    # Prepare arguments for runs
    runArgs = None

    # Have rank 0 prep args and broadcast
    if mpi_rank == 0:
        runArgs = target_prep_cmd_params( tInfo, cmdArgs )
        runArgs.setArg( 'newAll', True )
        runArgs.setArg( 'newInfo', True )
        
        # Broadcast master runArgs to everyone.
        mpi_comm.bcast( runArgs, root=0 )

    else:
        # Workers expecting runArgs.
        runArgs = mpi_comm.bcast( runArgs, root=0 )
    
    # Everyone wait
    mpi_comm.Barrier()
    
    # Have all verify if they have valid arguments
    if type(runArgs) == type(None):
        print("WARNING: SIMR.target_test_new_gen_scores: Invalid run arguments")
        return None
    
    # Create Workers and have them on standby to create scores
    if mpi_rank != 0:
        new_score_worker(tInfo, runArgs).run()
       
    #####################################
    #####   Variable Preperation   ######
    #####################################
       
       


        
def target_test_new_gen_scores( cmdArgs, tInfo ):
    
    printBase = tInfo.printBase
    printAll = tInfo.printAll
    
    if printBase and mpi_rank == 0:
        print( "SIMR.target_test_new_gen_scores: %d nodes" %( mpi_size ) )
    
    # Make sure you're in an MPI_environment with more than 1 core.
    if mpi_size == 1:
        print("WARNING: SIMR.target_test_new_gen_scores:")
        gm.tabprint("Must run code in mpirun with more than 1 core to work properly")
        gm.tabprint("Example: 'mpirun -n 4 python3 main_simr.py -newGen'")
        return
    
    # Prepare arguments for runs
    runArgs = None

    # Have rank 0 prep args and broadcast
    if mpi_rank == 0:
        runArgs = target_prep_cmd_params( tInfo, cmdArgs )
        
        # Add args for new score
        if type( runArgs ) != type( None ):
            runArgs.setArg( 'newAll', True )
            runArgs.setArg( 'newInfo', True )
            
        mpi_comm.bcast( runArgs, root=0 )

    # If in mpi and expecting info.
    else:
        runArgs = mpi_comm.bcast( runArgs, root=0 )
    
    # Everyone wait
    mpi_comm.Barrier()
    
    # Have all verify if they have valid arguments
    if type(runArgs) == type(None):
        print("WARNING: SIMR.target_test_new_gen_scores: Invalid run arguments")
        return None
    
    zScores, mData = (None, None)
    
    # Have Master create queue and launch workers
    if mpi_rank == 0:
        
        print("WORKING: SIMR.target_test_new_gen_scores")
        gm.tabprint("Using zoo merger data for testing")
        
        # WORKING: Use zoo merger data for testing and developing.
        zScores, mData = tInfo.getOrbParam()
        newData = mData[0:2,:]
        
        # Create Master Class.
        master = Master( range(1, mpi_size ) )
         
        # Create Queue with Master class
        mpi_queue = WorkQueue( master )
        
        # Create first queue of items
        # Add data to compute in queue
        for i in range( newData.shape[0] ):
            mpi_queue.add_work( data=( i, newData[i,:] ) )
        
        # Stay in loop until queue has items
        while not mpi_queue.done():
            
            # If workers are available, tell worker to do work.
            mpi_queue.do_work()  # Linked to class new_score_worker
            
            # Get return data if available
            for return_data in mpi_queue.get_completed_work():
                run_i, message = return_data
                print("Master received: %d - %s" % (run_i, message) )
                    
            pass
        
        # Create second queue of items
        # Add data to compute in queue
        for i in range( 15, 17 ):
            mpi_queue.add_work( data=( i, mData[i,:] ) )
        
        # Stay in loop until queue has items
        while not mpi_queue.done():
            
            # If workers are available, tell worker to do work.
            mpi_queue.do_work()  # Linked to class new_score_worker
            
            # Get return data if available
            for return_data in mpi_queue.get_completed_work():
                run_i, message = return_data
                print("Master received: %d - %s" % (run_i, message) )
                    
            pass
        
        master.terminate_slaves()
    
    else:        
        # Create worker class and be ready to accept work.
        new_score_worker(tInfo, runArgs).run()
        
        
def target_initialize( cmdArg=gm.inArgClass(), tInfo = None ):
    
    tDir = cmdArg.targetDir
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
    
    if cmdArg.printAll: 
        cmdArg.printBase = True

    if printBase:
        print("SIMR.target_initialize:")
        print("\t - tDir: %s" % tDir )
        print("\t - tInfo: %s" % type(tInfo) )

    # Check if given a target
    if tInfo == None and tDir == None and cmdArg.get('tInfo') == None:
        print("WARNING: SIMR.target_initialize:")
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
            gm.tabprint("%s - %s - %d Models" % ( tInfo.status, tInfo.get('target_id'), len( tInfo.tDict['zoo_merger_models'] ) ) )

    # Check if valid directory
    if tInfo.status == False:
        print("WARNING: SIMR.target_initialize:  Target Info status bad")
        return None
            
    if cmdArg.get('printParam', False):
        tInfo.printParams()
        
    cmdArg.setArg('tInfo', tInfo)
    
    
    # Check if valid score parameters.
    params = cmdArg.get('scoreParams')
    paramName = cmdArg.get('paramName',None)
    paramLoc = gm.validPath( cmdArg.get('paramLoc',None) )
    
    # If params there, move on
    if params != None:
        pass
    
    # If invalid, complain
    elif params == None and paramLoc == None and paramName == None:
        if printBase:
            print("WARNING: target_initialize: params not valid")
            gm.tabprint('ParamType: %s'%type(params))
            gm.tabprint('ParamName: %s'%paramName)
            gm.tabprint('ParamLoc : %s'%paramLoc)

    # If given a param name, assume target knows where it is.
    elif params == None and paramName != None:
        params = tInfo.readScoreParam(paramName)
    
    # If given param location, directly read file
    elif paramLoc != None:
        params = gm.readJson(paramLoc)

    # Check for final parameter file is valid
    if params == None:
        if printBase:
            print("WARNING: SIMR.target_initialize: Failed to load parameter class")    
    
    cmdArg.setArg('scoreParams', params)
    
    # Check if needing to make any images for new scoring.
    
    if cmdArg.get( 'newTargetImage', False ):
        if printBase:
            print("SIMR.target_initialize: Creating new target image")
        
        if type( params ) == type( None ):
            print("WARNING: SIMR.target_initialize: Please provide params for new target image.")
        
        # Loop through all params for creation
        for key in params:
            ic.adjustTargetImage( tInfo = tInfo, new_param = params[key], \
                                 overWrite = cmdArg.get('overWrite'), printAll = printAll )           
    return tInfo

# End target_initialize


def target_prep_cmd_params( tInfo, tArg ):

    printBase = tArg.printBase
    printAll = tArg.printAll
    
    if printBase:
        print("SIMR: target_prep_cmd_params: %s" % tInfo.get('target_id') )


    
    if tInfo.printAll:
        gm.tabprint( '%d in target_prep_cmd_params: %s: '% (mpi_rank, tInfo.get('target_id') ) ) 
    
    # Copy cmd Arg to send to runs
    runArgs = deepcopy( tArg )
    
    # Change printing variables
    if tArg.get('printAllRun',False):
        runArgs.setArg('printBase',True)
        runArgs.setArg('printAll',True)  
        
    elif tArg.get('printBaseRun',False):
        runArgs.setArg('printBase',True)
        runArgs.setArg('printAll',False)     
    
    # Default is to not have all 1000's of runs print something
    else:
        runArgs.setArg('printBase',False)
        runArgs.setArg('printAll',False)
    
    if printAll:
        print('SIMR.target_prep_cmd_params: Printing new run argments\n')
        runArgs.printArg()
        print('')
    
    return runArgs
        

def target_new_scores( tInfo, tArg ):
    
    runArgs = None
    
    # If not in MPI Env
    if mpi_size == 1:
        runArgs = target_prep_cmd_params( tInfo, tArg )
    
    # In MPI_Env
    else:
        
        # Have rank 0 prep args and broadcast
        if mpi_rank == 0:
            
            runArgs = target_prep_cmd_params( tInfo, tArg )
            mpi_comm.bcast( runArgs, root=0 )

        # If in mpi and expecting info.
        else:
            runArgs = mpi_comm.bcast( params, root=0 )

        mpi_comm.Barrier()
    
    if type(runArgs) == type(None):
        print("WARNING: SIMR.target_new_scores: Invalid run arguments")
        return None


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
