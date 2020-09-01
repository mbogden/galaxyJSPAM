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

    rInfo = im.run_info_class( runDir=rDir, printAll=printAll )

    if printAll:
        print('PT: rInfo.status: ', rInfo.status )

    #rInfo.printInfo()

# end processing run dir


# Process target directory
def procTarget( tDir, printAll=False ):

    tInfo = im.target_info_class( targetDir=tDir, printAll=True )

    for run in runDirs:
        #procRun( rDir )
        break

# End processing sdss dir

def procAllData( dataDir, printAll=False ):
    if arg.printAll: print("PT: dataDir: %s" % arg.dataDir )

    # Check if directory exists
    if not path.exists( dataDir ):  
        print("PT: WARNING: Directory not found")
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

    print( 'PT.procAllData: ')
    print( '\t - Found Dir: %d' % len( tDirList ) )

# Run main after declaring functions
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
