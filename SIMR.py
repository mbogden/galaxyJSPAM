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


    if arg.printAll:
        print("SIMR: Hi!  In Matthew's code for connecting various pieces of code.")
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
        procAllData( arg.dataDir, printAll=arg.printAll, targetFile = getattr( arg, 'targetFile', None)  )

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

    '''
    for run in runDirs:
        rDir = gDir + run + '/'
        procRun( rDir )

    '''

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


def procAllData( dataDir, targetFile=None, printBase=True, printAll=False, printProg=True ):

    if printAll: 
        print("PT: procAllData:")
        print("\t - dataDir: %s" % dataDir )
        print("\t - printAll: ", printAll )


    targetDirList = listdir( dataDir )
    targetDirList.sort()

    if targetFile != None:
        print("SIMR: Filtering targets")

        if not path.exists( targetFile ):
            print("SIMR: WARNING: targetFile not found: %s" % targetFile)

        tFile = gm.readFile( targetFile, stripLine=True )

        tList = [ t for t in targetDirList if t in tFile ]
        
        print('\t- Before: %d' % len( targetDirList) )
        print('\t- After: %d' % len( tList) )

        targetDirList = tList

    # End filtering based on targets of interest

    nTargets = len( targetDirList )

    if printAll:
        print("SIMR: Found target directories: %d" % nTargets )

    # Gather up information data on interested targets
    tInfoList = []
    for tName in targetDirList:
        tDir = dataDir + tName + '/'

        if printAll: 
            print('\t - target: %s (%s) "%s"' % ( tName, path.exists(tDir), tDir ) )

        tInfo = im.target_info_class( targetDir = tDir, printAll = printAll, rmInfo=False )
        tInfo.updateProgress()
        
        if tInfo.status:
            tInfoList.append( tInfo )

        else:
            print("SIMR: ERROR!: target failed to read info file")
            print('\t- targetDir: %s (%s) %s' % (tName, path.exists(tDir), tDir ))
            return

    if printProg:
        checkDataProg( tInfoList )


def checkDataProg( tInfoList ):

    n = len( tInfoList )

    pDict = {}
    
    for key in tInfoList[0].progHeaders:
        pDict[key] = 0

    for tInfo in tInfoList:

        pInfo = tInfo.tDict['progress']

        for key in tInfo.progHeaders:

            if pInfo.get( key, None ) == None:
                continue

            # If true or false value
            elif type( pInfo[key] ) == type( True ):
                if pInfo[key] == True:
                    pDict[key] += 1

            elif type( pInfo[key] ) == int:
                pDict[key] += 1
                    
            # If string value with quantities
            elif type( pInfo[key] ) == type( 'string' ):

                from re import findall
                
                # Check for quantities is given
                numList = findall( pInfo[key], '/d+' )
                print(numList)
                
                # If just string or single number given
                if len( numList ) == ( 1 or 0 ):
                    pDict[key] += 1

                # two quantities given (some/total)
                elif len( numList ) == 2 and '/' in pInfo[key]:
                    n2, n3 = pInfo[key].split('/')
                    if n2 == n3:
                        pDict[key] += 1


    print("SIMR: All Target Progression.")
    for key in pDict:
        print( '\t- %s : %d/%d (%s)' % ( key, pDict[key], n, pDict[key] == n ) )
    

# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
