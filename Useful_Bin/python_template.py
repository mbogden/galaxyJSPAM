'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    29 Oct 2019
Description:    This is my python template for how I've been making my python programs
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


class inArgClass:

    def __init__( self, inArg=None ):
        self.test = 'test val'
        self.dataDir = None
        self.sdssDir = None
        self.runDir = None
        
        if inArg != None:
            self.updateArg( inArg )

    def updateArg( self, inArg ):

        n = len( inArg )

        # Loop through given arguments
        for i, arg in enumerate( inArg ):

            # Ignore unless handle provided
            if arg[0] != '-':
                continue

            argName = arg[1:]

            # Check if last arg
            if i+1 == n:
                argVal = True

            # Check if suplimentary info provided
            elif inArg[i+1][0] != '-':
                argVal = inArg[i+1]

            # If no supplimentary arg given, assume True
            else:
                argVal = True

            setattr( self, argName, argVal )

    # End update input arguments

    def printAllVar(self):

        allAttrs = vars( self )

        print("All attributes of " , self)
        for a in allAttrs:
            print('\t- %s :' % a, getattr(self, a ) )



# Run main after declaring functions
if __name__ == '__main__':
    argList = argv
    main(argList)
