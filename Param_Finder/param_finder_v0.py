'''
    Author:     Matthew Ogden
    Created:    31 July 2019
    Altered:    
Description:    This code tries to find the optimal image parameters for creating an image
                    out of SPAM data and a comparison method.
'''

from sys import \
        exit, \
        argv

from os import \
        listdir, \
        path, \
        system

import numpy as np
import multiprocessing as mp


# Global constants
printAll = True

parProc = True
nProc = 1

runDir = ''
paramDir = ''
makeDir = False

saveParamDir = 'Input_Data/image_parameters/'
paramName = 'tparam_v0_0000.txt'

imgCreatorLoc = 'Image_Creator/python/image_creator_v2'


def main():

    endEarly = readArg()

    if printAll:
        print('Param directory: %s' % paramDir)
        print('runDir: %s' % runDir)
        print('Beginning Image Parameter Finder')
        print('Requesting cores: %d' % nProc)
        print('Number of cores available: %s' % mp.cpu_count())

    if endEarly:
        print('Exiting...')
        sys.exit(-1)

    # Make parameter directory if needed
    if makeDir:
        makeParamDir(paramDir,runDir)

    # Create initial parameter
    p = imageParameterClass()
    if printAll:
        print("Starting image values")
        p.printVal()

    notDone = True
    while( notDone ):

        createImg_v2(p)

        notDone = False
    
    #p.writeParam(saveParamDir + paramName)

# End main


def createImg_v2(p):
    if printAll:
        print('In create Image function')



def makeParamDir( paramDir, runDir ):

    if printAll:
        print('Making paramDir: %s' % paramDir)
        print('Using run dir: %s' % runDir)

    if not path.exists( runDir ):
        print('Could not find run directory to copy')
        exit(-1)

    system('mkdir -p %s' % paramDir )
    system('cp %s* %s.' % ( runDir, paramDir) ) 

    if printAll:
        print('Made directory: %s' % paramDir)


def testPrint(arg1, arg2):

    myRank = mp.current_process()._identity[0]
    print('Began - %s' % ( myRank))

# End testPrint



def readArg():

    global printAll, nProc, parProc
    global runDir, makeDir, paramDir

    argList = argv
    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir += '/'

        elif arg == '-newParamDir':
            makeDir = True

        elif arg == '-paramDir':
            paramDir = argList[i+1]
            if paramDir[-1] != '/':
                paramDir += '/'
 
        elif arg == '-pp':
            nProc = argList[i+1]


    # End looping through command line arguments

    # Check if input arguments were valid

    # Check if number of processors is practical
    try:
        nProc = int( nProc )

    except:
        print('WARNING: numbers of cores requested not a number. \'%s\'' % nProc )
        endEarly = True

    else:
        if nProc > 1:
            max_CPU_cores = int(mp.cpu_count())

            if nProc > max_CPU_cores:
                print('WARNING:  Number of cores requested is greater than the number of cores available.')
                print('Requested: %d' % nProc)
                print('Available: %d' % max_CPU_cores)
                endEarly = True

    # Check if paramDirectory exists
    if not makeDir and paramDir == '':
        print('No paramDirectory given.')
        endEarly = True

    # Check if making a new directory
    if makeDir and runDir == '':
        print('Please specify inital runDir if making a new param finding directory')
        endEarly = True

    return endEarly

# End reading command line arguments


def readArgFile(argList, argFileLoc):

    try:
        argFile = open( argFileLoc, 'r')
    except:
        print("Failed to open/read argument file '%s'" % argFileLoc)
    else:

        for l in argFile:
            l = l.strip()

            # Skip line if comment
            if l[0] == '#':
                continue

            # Skip line if empty
            if len(l) == 0:
                continue

            lineItems = l.split()
            for item in lineItems:
                argList.append(item)
        # End going through arg file 

    return argList

# end read argument file


# Define image parameter class
class imageParameterClass:

    def __init__(self):
        self.name    = 'blank'
        self.gSize   = int(25)     # gaussian size
        self.gWeight = 5         # gaussian size
        self.rConst  = 5         # radial constant
        self.bConst  = 5         # birghtness constant
        self.nVal    = 5         # normalization constant
        self.nRow    = int(400)   # number of rows
        self.nCol    = int(600)   # number of col
        self.gCenter = np.array( [[ 200, 400 ],      # [[ x1, x2 ] 
                             [ 200, 200 ]])     #  [ y1, y2 ]]
        self.comment = 'blank comment'

        # Step size for numerical derivative
        self.h_gSize = 1
        self.h_gWeight = 0.05
        self.h_rConst = 0.05
        self.h_bConst = 0.05
        self.h_nVal = 0.05
    # end init

    def printVal(self):
        print(' Name: %s' % self.name)
        print(' Comment: %s' % self.comment)
        print(' Gaussian size: %d' % self.gSize)
        print(' Gaussian weight: %f' % self.gWeight)
        print(' Radial constant: %f' % self.rConst)
        print(' Brightness constant: %f' % self.bConst)
        print(' Normalization constant: %f' % self.nVal)
        print(' Number of rows: %d' % self.nRow)
        print(' Number of columns: %d' % self.nCol)
    # end print

    def writeParam(self, saveLoc):
        try:
            pFile = open(saveLoc,'w')
        except:
            print('Failed to create: %s' % saveLoc)
        else:
            pFile.write('parameter_self.name %s\n' % self.name)
            pFile.write('comment: %s\n\n' % self.comment)
            pFile.write('gaussian_size %d\n' % self.gSize)
            pFile.write('gaussian_weight: %f\n' % self.gWeight)
            pFile.write('radial_constant: %f\n' % self.rConst)
            pFile.write('brightness_constant: %f\n' % self.bConst)
            pFile.write('norm_constant: %f\n' % self.nVal)
            pFile.write('image_rows: %d\n' % self.nRow)
            pFile.write('image_cols: %d\n' % self.nCol)
            pFile.write('galaxy1_center: %f %f\n' % ( self.gCenter[0,0], self.gCenter[0,1] ))
            pFile.write('galaxy2_center: %f %f\n' % ( self.gCenter[1,0], self.gCenter[1,1] ))
            pFile.close()
# End parameter class

main()

# Example on how to begin a parrel environment
'''
processes = []
arg1 = 'temp_arg'
arg2 = 15

# Start all processes
for i in range( nProc ):
    p = mp.Process( target=testPrint, args=( arg1, arg2, ) )
    processes.append( p )
    p.start()

# Wait until all processes are complete
for p in processes:
    p.join()
'''

