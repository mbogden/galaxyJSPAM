#!/usr/bin/env python
# coding: utf-8

# In[20]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author:	Matthew Ogden
    Created:	22 Feb 2022
Description:	Primary code for using pre-built Neural Networks.
'''

# Python module imports
import os, numpy as np, tensorflow as tf
from sys import path as sysPath




# ## Check if I'm in a Jupyter environment for the initial writing and modification of the program.

# In[33]:


# Am I in a jupyter notebook?
try:

    # Test if in jupyter
    get_ipython().__class__.__name__
    buildEnv = True
    print ("DNN: In Building Environment")

    # Currect directory should be location this file is in
    __file__ = os.getcwd()
    print( 'File Loc: ',  __file__ )

# Or am I in a python script?
except:
    if buildEnv: print("Build Failed")
    pass


# ## Load Custom Modules, Functions, and Global variables for the galaxyJSPAM suite

# In[36]:


# For loading in Matt's general purpose python libraries
sysPath.append( path.abspath( "Support_Code/" ) )
sysPath.append( path.abspath( path.join( __file__ , "../Support_Code/" ) ) )
import general_module as gm
import info_module as im

def test():
    print("DNN: Hi!  You're in Matthew's main program for neural networks.")

class score_nn_model:
    modelLoc = None
    model = None

    def __init__(self, modelLoc):
        self.modelLoc = modelLoc
        self.load_model( modelLoc )

    def load_model(self, modelLoc):
        self.model = tf.keras.models.load_model( modelLoc )

    def score(self, data):
        return self.model.predict( data )

if buildEnv: 
    test()

    # Set example input arguements
    args = gm.inArgClass()
    args.setArg( 'printAll', True )
    args.setArg( 'targetDir', '~/galStuff/data/')
    args.printArg()


# ## Main Fuction

# In[ ]:


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


# 
