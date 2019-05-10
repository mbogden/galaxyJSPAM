'''
Author:     Matthew Ogden
Date:       10 May 2019
Purpose:    This program is meant to perform a runs of the SPAM software based on Zoo model files
'''

from sys import (
        argv,
        exit
        )

from os import (
        path,
        system
        )

# Default Global Variables

interactiveMode = False     # For future implementations

# Incomplete implemenation
useParProc = False
nProc = 1

basicRun = 'SPAM/bin/basic_run'

zooFile = ''
zooDir = 'Input_Data/zoo_models'

outputDir = ''
addRunDir = True   #Create run directory to save particle files in

nZooRuns = 0
allZooRuns = False  # All zoo runs in file

maxN = 10000000     # 10 Mil, can be changed
nPart = 10000       # default 10k


def main():

    global nZooRuns

    readCommandLine()

    # Print info being used
    print('Using SPAM basic_run: \'%s\'' % basicRun)
    print('Using Zoo Model File: \'%s\' ' % zooFile )

    if nZooRuns > 0:
        print('Using %d models from zoo file' % nZooRuns)

    if allZooRuns:
        print('Using all major models from Galaxy Zoo file')
        nZooRuns = float("inf")     # set number of run to infinite to do all

    print('Using %d particles in basic_run' % nPart)

    # Read Galaxy zoo file
    modelList = readZooFile(zooFile, nZooRuns)

    # Begin generating runs
    zooName = zooFile.split('/')[-1]
    zooName = zooName.split('.')[0]
    print('Using name %s' % zooName)

    runBasicRuns( zooName, nPart, modelList )

# End main


def runBasicRuns( zooName, nPart, modelList ):
    print( 'In run basic runs')

    for i,data in modelList:
        sysCmd = './%s -m %d -n1 %d -n2 %d %s' % ( basicRun, i, nPart, nPart, data )
        print(sysCmd)
        system(sysCmd)
# End runBasicRuns


def readZooFile( zooFile, nZooRuns):
    zFile = open(zooFile,'r')
    print('Reading Galaxy Zoo file...')
    fileData = []
    count = 1

    # Loop line by line through file
    for i,l in enumerate(zFile):

        # break out of file loop if found rejected line
        if '#rejected' in l:
            break

        # break out of file loop if reached number of desired runs
        if i >= nZooRuns:
            break
        
        spamData = l.strip().split('\t')[1]
        
        nData = len(spamData.split(','))

        # ensure grabbed data is of proper format
        if nData == 34:
            fileData.append([count,spamData])
            count += 1
        else:
            print("#######################    POSSIBLE ERROR   ########################")
            print('Found spam data in file \'%s\' on line %d, \'%s\' that does not match default 34 values seperated by a \',\' after a tab' % ( zooFile, i, l.strip() ) )

    # End looping through file

    print(fileData[-1][0], ' models found')

    zFile.close()

    return fileData
# End readZooFile()


def readCommandLine():

    global useParProc, nProc, basicRun, nPart, outputDir
    global zooDir, zooFile, allZooRuns, nZooRuns

    for i,arg in enumerate(argv):

        if arg == '-0':
            print('You found Zero')

        # Define Number of processors
        elif arg == '-p':
            useParProc = True
            try:
                nProc = int(argv[i+1])
            except:
                print('\'%s\' is an invalid number for number of processors' %  argv[i+1] )
                exit(-1)

        # Define range of 
        elif arg == '-r':
            basicRun = argv[i+1]

        # Define basic run location
        elif arg == '-br':
            basicRun = argv[i+1]

        # Define Zoo Model File
        elif arg == '-zf':
            zooFile = argv[i+1]

        # Define directory with zoo models
        elif arg == '-zd':
            zooDir = argv[i+1]

        # Do N number of runs in zoo model file
        elif arg == '-zn':
            try:
                nZooRuns = int(argv[i+1])
            except:
                print('\'%s\' is an invalid number for the number zoo models' %  argv[i+1] )
                exit(-1)

        # Do all Zoo models (All in competition, not all in file)
        elif arg == '-za':
            allZooRuns = True

        # Specify the number of particles in each galaxy
        elif arg == '-n':
            try:
                nPart = int(argv[i+1])
            except:
                print('\'%s\' is an invalid number for the number of particles' %  argv[i+1] )
                print('Exiting...\n')
                exit(-1)
        # end '-n'
        
        # Check if wants a run directory or not
        elif arg == '-nr':
            makeRunDir = False

        # Output Directory
        elif arg == '-o':
            outputDir = argv[i+1]

    # end looping through arguments

    # Make checks on input data
    exitProgram = False

    # Is number good? 
    if nPart <= 0:
        print('Number of particles must be greater than 0')
        exitProgram = True

    elif nPart > maxN:
        print('Not accepting number of particle greater than %d' % maxN)
        exitProgram = True

    # Is basic run an actual file? 
    isFile = path.isfile(basicRun)
    if not isFile:
        print('Basic Run executable at \'%s\' was not found' % basicRun)
        exitProgram = True

    if nZooRuns == 0 and allZooRuns == False:
        print('Please specify how many runs you would like from zoo file')
        exitProgram = True

    if nZooRuns > 0 and allZooRuns == True:
        print('Please specify either all Zoo runs or a Number of zoo runs')
        exitProgram = True

    isFile = path.isfile(zooFile)
    if not isFile:
        print('Galaxy Zoo model file \'%s\' not found' % zooFile)
        print('Please specify by adding \'-zf zooDirectory/zooFile.txt\'')
        exitProgram = True

    # Exit program with error
    if exitProgram:
        print('Exiting...\n')
        exit(-1)

# End readCommandLine()


# Execute everything after declaring functions
print('')
main()
print('')
