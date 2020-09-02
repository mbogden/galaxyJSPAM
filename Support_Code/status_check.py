'''
    Author:     Matthew Ogden
    Created:    02 Apr 2020
Description:    This is my code for checking the status of spam data/models/images/etc
'''

from os import path, listdir
from sys import exit, argv, path as sysPath

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Suuport_Code/" ) )
sysPath.append( supportPath )
import general_module as gm
import info_module as im

def test():
    print("ST: Hi!  In Matthew's python template for creating new SPAM related code.")
    gm.test()
    im.test()

def main(argClass):
    print("ST: Hi!  In Matthew's python template for creating new SPAM related code.")
    argClass.printArg()

    if argClass.runDir != None:
        print('cool')

        im.updateRunInfo( runLoc )

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