'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
Description:    This is my python template for how I've been making my python programs
'''

from os import path, listdir
from sys import exit, argv, path as sysPath

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Suuport_Code/" ) )
sysPath.append( supportPath )
import general_module as gm


def main(argList):
    print("Hi!  In main python template")

# End main

def procSdss( sDir ):
    iDir = sDir + 'information/'
    gDir = sDir + 'gen000/'
    pDir = sDir + 'plots/'
    scoreDir = sDir + 'scores/'

    if not path.exists( iDir ) or not path.exists( gDir ):
        print("Somethings wrong with sdss dir")

    runDirs = listdir( gDir )
    runDirs.sort()

    for run in runDirs:
        rDir = gDir + run + '/'
        procRun( rDir )
# End processing sdss dir


def procRun( rDir ):
    modelDir = rDir + 'model_images/'
    miscDir = rDir + 'misc_images/'
    ptsDir = rDir + 'particle_files/'
    infoLoc = rDir + 'info.txt'

    if not path.exists( modelDir ) or not path.exists( ptsDir ) or not path.exists( infoLoc):
        print("Somethings wrong with run dir")


# end processing run dir


# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
