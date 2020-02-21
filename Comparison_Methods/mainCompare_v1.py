'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    21 Feb 2020
Description:    This is my new attempt at having a comparison program for implementing comparison methods.
'''

from sys import \
        exit, \
        argv

from os import \
        path, \
        listdir

printAll = True
sdssDir = None

def main(argList):

    endEarly = readArg(argList)

    if endEarly:
        exit(-1)

    if sdssDir != None:
        procSdss( sdssDir )

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


def readArg(argList):

    global printAll, sdssDir

    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]
            if sdssDir[-1] != '/': sdssDir += '/'

    # Check if input arguments are valid
    if sdssDir != None and not path.exists( sdssDir ):
        print("Sdss dir is not a path: %s" % sdssDir )

    return endEarly

# End reading command line arguments


def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return None
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return None

    else:
        inList = list(inFile)
        inFile.close()
        return inList

# End simple read file

# Run main after declaring functions
if __name__ == '__main__':
    argList = argv
    main(argList)
