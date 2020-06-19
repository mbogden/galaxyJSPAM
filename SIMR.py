'''
    Author:     Matthew Ogden
    Created:    22 Apr 2020
Description:    *** Initial Code Creation ***
                This is the almighty main code that calls all the other code for spam
'''

from os import path, listdir
from sys import exit, argv, path as sysPath

# For loading in Matt's general purpose python libraries
#supportPath = path.abspath( path.join( __file__ , "../Support_Code/" ) )
#sysPath.append( supportPath )
import Support_Code.general_module as gm
import Support_Code.info_module as im


def main(arg):

    print("Hi!  In Matthew's python template for creating new SPAM related code.")

    if arg.printAll:
        gm.test()
        im.test()
        arg.printArg()

    # end main print
    
    if arg.simple:
        if arg.printAll: print("PT: Simple!~")

    elif arg.runDir != None:
        procRun( arg.runDir, printAll=arg.printAll )

    elif arg.targetDir != None:
        procTarget( arg.targetDir, printAll=arg.printAll )

    elif arg.dataDir != None:
        procAllData( arg.dataDir, printAll=arg.printAll, zModelDir = getattr( arg, 'zModelDir', None)  )

    # If no main options selected
    else:
        print("PT: Nothing selected!")
        print("PT: Recommended options")
        print("\t -simple")
        print("\t -runDir /path/to/dir/")
        print("\t -targetDir /path/to/dir/")
        print("\t -dataDir /path/to/dir/")


# End main

# Process target directory
def procTarget( tDir, printAll=False ):

    if arg.printAll: print("PT: targetDir: %s" % tDir )

    if type(tDir) != type('string'):
        print("ERROR: PT: Target: targetDir not a string: %s - %s" % ( type(tDir), tDir ) )
        return False

    if not path.exists( tDir ):
        print("ERROR: PT: Target: targetDir not found: " % tDir )
        return False

    tInfo = im.target_info_class( targetDir = tDir )

    runDirs = listdir( gDir )
    runDirs.sort()

    for run in runDirs:
        rDir = gDir + run + '/'
        procRun( rDir )

# End processing sdss dir



def procRun( rDir, printAll=False, checkStatus=False ):

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


def procAllData( dataDir, printAll=False, zModelDir=None ):

    if printAll: 
        print("PT: procAllData:")
        print("\t - dataDir: %s" % dataDir )
        print("\t - printAll: ", printAll )
        print("\t - zModelDir: ", zModelDir )

    if zModelDir[-1] != '/': zModelDir += '/'

    dirList = listdir( dataDir )
    dirList.sort()

    if printAll:
        print("SIMR: Found target directories: %d" % len( dirList ) )

    for tName in dirList:
        tDir = dataDir + tName + '/'

        zModelLoc = zModelDir + tName + '.txt'
        if printAll: 
            print('\t - target: %s (%s) "%s"' % ( tName, path.exists(tDir), tDir ) )
            print('\t - modelFile: (%s) %s' % ( path.exists( zModelLoc ), zModelLoc ) )

        tInfo = im.target_info_class( targetDir = tDir, printAll = printAll, rmInfo=True )



# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
