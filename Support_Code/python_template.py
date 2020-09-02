'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
Description:    This is my python template for how I've been making my python programs
'''

from os import path
from sys import path as sysPath

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
sysPath.append( supportPath )
import general_module as gm
import info_module as im


def main(arg):

    print("PT: Hi!  In Matthew's python template for creating new SPAM related code.")

    if arg.printAll:

        arg.printArg()
        gm.test()
        im.test()
    # end main print
    
    if arg.simple:
        if arg.printBase: 
            print("PT: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None:
        procRun( arg.runDir, printAll=arg.printAll )

    elif arg.targetDir != None:
        procTarget( arg.targetDir, printAll=arg.printAll )

    elif arg.dataDir != None:
        procAllData( arg.dataDir, printAll=arg.printAll )

    else:
        print("PT: Nothing selected!")
        print("PT: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

# End main


def procRun( rDir, printBase=True, printAll=False ):

    if printBase:
        print("PT.procRun: Inputs")
        print("\t - rDir:", rDir)

    rInfo = im.run_info_class( runDir=rDir, printBase = False, printAll=printAll )

    if printBase:
        print('PT.procFun: rInfo.status: ', rInfo.status )

    if rInfo.status == False:
        print('PT.procRun: WARNGING:\n\t - rInfo status not good. Exiting...' )
        return

    if printAll:
        rInfo.printInfo()


# end processing run dir


# Process target directory
def procTarget( tDir, printBase = True, printAll=False ):

    if printBase:
        print("PT.procTarget:")
        print("\t - tDir: " , tDir)

    tInfo = im.target_info_class( targetDir=tDir, printAll=True )

    if printBase:
        print("PT.procTarget:")
        print("\t - tInfo.status: %s" % tInfo.status )

    # Check if target is valid
    if tInfo.status == False:
        print("PT: WARNING: procTarget:")
        print("\t - Target not good.")
        return

    if printAll:
        tInfo.printInfo()

    tInfo.gatherRunInfos()
    
    for r in tInfo.runDirs:
        print("t")

    

# End processing sdss dir

def procAllData( dataDir, printBase=True, printAll=False ):

    from os import listdir

    if printBase: 
        print("PT.procAllData")
        print("\t - dataDir: %s" % arg.dataDir )

    # Check if directory exists
    if not path.exists( dataDir ):  
        print("PT.procAllData: WARNING: Directory not found")
        print('\t - ' , dataDir )
        return

    # Append trailing '/' if needed
    if dataDir[-1] != '/': dataDir += '/'  

    dataList = listdir( dataDir )   # List of items found in folder
    tDirList = []  # List of folders that are target directories

    # Find target directories
    for folder in dataList:
        tDir = dataDir + folder
        tempInfo = im.target_info_class( targetDir=tDir, printAll=False )

        # if a valid target directory
        if tempInfo.status:  tDirList.append( tempInfo )

    if printBase:
        print( '\t - Target Directories: %d' % len( tDirList ) )

# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
