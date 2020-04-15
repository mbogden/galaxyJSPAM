'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
Description:    This is my python template for how I've been making my python programs
'''

from os import path, listdir
from sys import exit, argv, path as sysPath

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
sysPath.append( supportPath )
import general_module as gm
import info_module as im


def main(arg):

    print("Hi!  In Matthew's python template for creating new SPAM related code.")

    if arg.printAll:

        arg.printArg()
        gm.test()
        im.test()
    # end main print
    
    if arg.simple:
        if arg.printAll: print("PT: Simple!~")

    elif arg.runDir != None:
        procRun( arg.runDir, printAll=arg.printAll )

    elif arg.targetDir != None:
        procTarget( arg.targetDir, printAll=arg.printAll )

    elif arg.dataDir != None:
        procAllData( arg.dataDir, printAll=arg.printAll )

    else:
        print("PT: Nothing selected!")
        print("PT: Recommended options")
        print("\t -simple")
        print("\t -runDir /path/to/dir/")
        print("\t -targetDir /path/to/dir/")
        print("\t -dataDir /path/to/dir/")


# End main


def procRun( rDir, printAll=False ):

    if type(rDir) != type('string'):
        print("ERROR: PT: runDir not a string: %s - %s" % ( type(rDir), rDir ) )
        return False

    if not path.exists( rDir ):
        print("ERROR: PT: runDir not found: " % rDir )
        return False

    if arg.printAll: print("PT: runDir: %s" % arg.runDir )

    modelDir = rDir + 'model_images/'
    miscDir = rDir + 'misc_images/'
    ptsDir = rDir + 'particle_files/'
    infoLoc = rDir + 'info.json'

    if not path.exists( modelDir ) or not path.exists( ptsDir ) or not path.exists( infoLoc):
        print("ERROR: PT: A directory was not found in runDir")
        print("\t- modelDir: " , path.exists( modelDir ) )
        print("\t-  miscDir: " , path.exists( miscDir ) )
        print("\t-  partDir: " , path.exists( ptsDir ) )
        return False

    rInfo = im.run_info_class( runDir=rDir, printAll=printAll )
    #rInfo.printInfo()

# end processing run dir


# Process target directory
def procTarget( tDir, printAll=False ):

    if arg.printAll: print("PT: sdssDir: %s" % arg.sdssDir )

    if type(tDir) != type('string'):
        print("ERROR: PT: Target: targetDir not a string: %s - %s" % ( type(tDir), tDir ) )
        return False

    if not path.exists( tDir ):
        print("ERROR: PT: Target: targetDir not found: " % tDir )
        return False

    iDir = tDir + 'information/'
    gDir = tDir + 'gen000/'
    pDir = tDir + 'plots/'

    infoLoc = iDir + 'target_info.json'

    # If not a target folder
    if not path.exists( iDir ) or not path.exists( gDir ):
        print("ERROR: PT: Couldn't find needed folders in targetDir.")
        print("\t- infoDir: " , path.exists( iDir ) )
        print("\t-  genDir: " , path.exists( gDir ) )
        return False

    tInfo = im.target_info_class( infoLoc )

    runDirs = listdir( gDir )
    runDirs.sort()

    for run in runDirs:
        rDir = gDir + run + '/'
        procRun( rDir )

# End processing sdss dir

def procAllData( dataDir, printAll=False ):
    if arg.printAll: print("PT: dataDir: %s" % arg.dataDir )
    print("All data")

# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
