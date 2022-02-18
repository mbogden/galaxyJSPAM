#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author:	Matthew Ogden
    Created:	22 Feb 2022
Description:	Primary code for creating and using Neural Networks.
'''

# Python module imports
from os import path, listdir
from sys import path as sysPath
import pandas as pd 
import numpy as np 
import cv2 
from mpi4py import MPI 
mpi_comm = MPI.COMM_WORLD    
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


# For loading in Matt's general purpose python libraries
sysPath.append( path.abspath( "Support_Code/" ) )
sysPath.append( path.abspath( path.join( __file__ , "../../Support_Code/" ) ) )
import general_module as gm
import info_module as im

def test():
    print("NN: Hi!  You're in Matthew's main program for neural networks.")

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
            print( 'NN: main: In MPI environment!')
        sleep(1)
        if arg.printAll: gm.tabprint('I am %d of %d ' %( mpi_rank, mpi_size ) ) 
        mpi_comm.Barrier()
        
    if arg.printAll and mpi_rank == 0:
        arg.printArg()
        gm.test()
        im.test()
        fe.test()
        ms.test()
        ss.test()

    # end main print

    if arg.simple:
        if arg.printBase and mpi_rank == 0: 
            print("NN: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None and mpi_rank == 0: 
        print("NN: nn_run not available.") 

    elif arg.targetDir != None: 
        simr_target( arg )

    elif mpi_rank == 0:
        print("NN: main: Nothing selected!")
        print("NN: main: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")

# End main

def target_new_neural_network( tArg, tInfo ):
    
    print('\n' + "*****"*5 + '\n')
    print("NN: WORKING: Creating 'target_new'neural_network")

    printBase = tArg.get('printBase',False)
    printAll = tArg.get('printAll',False)
    
    if printBase:
        print("NN: target_new_neural_network")
    
    if tArg.get('trainNNLoc',None) == None:
        print("NN: WARNING: Please provice -trainNNLoc to continue\n")
        return   
    
    trainNN = {}
    trainNN['name'] = "sl_test"
    trainNN['top_models'] = 500
    gm.saveJson( trainNN, tArg.get('trainNNLoc') )
    
    trainDict = gm.readJson( tArg.get('trainNNLoc') )
    if trainDict != None:
        print("Yay")
    
    
    print('\n' + "*****"*5 + '\n')
    

# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    
    main( arg )
