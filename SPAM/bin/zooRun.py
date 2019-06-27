'''
Author:     Matthew Ogden
Created:    10 May 2019
Altered:    25 Jun 2019
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

printAll = False

basicRun = 'SPAM/bin/basic_run'
uniqID = 0

outputDir = ''
makeRunDir = True   #Create run directory to save particle files in
compressFiles = False
overWrite = False


maxN = 10000000     # Arbitrary limit, can be changed if needed
nPart = 10000       # default 10k

sdssName = '00'
runNum = 0000
genNum = 00

useZooFile = False
zooFile = ''
zooDir = 'Input_Data/zoo_models'

nZooRuns = 0
allZooRuns = False  # All zoo runs in file

useModelData = False
modelData = ''



def main():

    readCommandLine()

    # Print info being used

    if printAll:
        print('Using SPAM basic_run: \'%s\'' % basicRun)
        print('Using %d particles in basic_run' % nPart)

        if useModelData:
            print('Using model data\'%s\'' % modelData)

        if useZooFile:
            print('Using Zoo Model File: \'%s\' ' % zooFile )

            if nZooRuns > 0:
                print('Using %d models from zoo file' % nZooRuns)

            if allZooRuns:
                print('Using all major models from Galaxy Zoo file')
                nZooRuns = float("inf")     # set number of run to infinite to do all


    # begin Execution

    if useModelData:
        uID = uniqID

        # Prep to move to directory
        if makeRunDir:
            oDir = outputDir + 'run_' + str(genNum).zfill(2) + '_' + str(runNum).zfill(4) + '/'
        else:
            oDir = outputDir

        if not overWrite and path.exists(oDir):
            print('%d : Directory already exists, exiting' % ( uniqID))
            return 1

        partCreated = runBasicRun( nPart, uID, modelData )

        if partCreated:
            partMoved = movePartFiles( uID, oDir )

        if partMoved:
            createInfoFile(oDir)

    # End if useModelData


    if useZooFile:

        # Read Galaxy zoo file
        modelList = readZooFile(zooFile, nZooRuns)

        # Begin generating runs
        sdssName = zooFile.split('/')[-1]
        sdssName = zooName.split('.')[0]
        if printAll:
            print('Using name %s' % zooName)

        # Changed runBasicRuns, recreate before using
        #runBasicRuns( zooName, nPart, modelList )

# End main


def createInfoFile( oDir ):

    infoFile = open(oDir + 'info.txt', 'w')

    infoFile.write('SPAM Particle Information\n')
    infoFile.write('sdss_name %s\n' % sdssName )
    infoFile.write('generation %d\n' % genNum )
    infoFile.write('run_number %d\n' % runNum )
    infoFile.write('g1_num_particles %d\n' % nPart)
    infoFile.write('g2_num_particles %d\n' % nPart)
    infoFile.write('\n')
    
    infoFile.close()
# end Create Info File


def movePartFiles( uID, oDir ):
 
    # Check if particle files were created
    fileNames = [ 'a_%d.000' % uID, 'a_%d.101' % uID]
    if not ( path.isfile(fileNames[0]) and path.isfile(fileNames[1])):
        print('###  WARNING  ###')
        print('Particle files not found')
        return False

    # Create directory if it doesn't exist
    if not path.exists(oDir):
        createCmd = 'mkdir -p %s' % oDir
        try:
            if printAll:
                print('Creating directory %s' % oDir)
            system(createCmd)
        except:
            print('Failed to create new directory ',createCmd)
            return False
    # End create dir

    # Zip files if requested
    if compressFiles:
        try:
            print('Zipping files')
            zipCmd1 = 'zip a_%d_000.zip a_%d.000' % (uID, uID)
            zipCmd2 = 'zip a_%d_101.zip a_%d.101' % (uID, uID)

            system(zipCmd1)
            system(zipCmd2)

            print('Moving files')
            mvCmd = 'mv a_%d_*.zip %s' % (uID, oDir)

            system(mvCmd)

            rmCmd = 'rm a_%d.*' % uID
            system(rmCmd)

        except:
            print('Failed zipping and moving')

    else:

        # Move particle files to output
        try:
            mvCmd = 'mv a_%d* %s' % ( uID, oDir ) 
            system(mvCmd)

        except:
            print('###  WARNING  ###')
            print('Could not move spam particle files... ',fileNames)
            return False
    
    return True

# movePartFiles



def runBasicRun( nPart, uID, data ):


    sysCmd = './%s -m %d -n1 %d -n2 %d %s' % ( basicRun, uID, nPart, nPart, data )
    if printAll:
        print('Running command: ',sysCmd)
    try:
        retVal = system(sysCmd)   # Consider implementing a way to check return val from spam code
        if printAll:
            print('Command Complete')
        return True

    except:
        print('Failed running command: ',sysCmd)
        return False

# End runBasicRun



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

    global basicRun, nPart, outputDir, makeRunDir
    global sdssName, runNum, genNum, uniqID, compressFiles
    global useZooFile, zooDir, zooFile, allZooRuns, nZooRuns
    global useModelData, modelData, printAll, overWrite

    for i,arg in enumerate(argv):

        if arg == '-0':
            print('You found Zero')

        # Define if you want printing
        elif arg == '-print':
            printAll = True

        # Define spam executable location
        elif arg == '-spam':
            basicRun = argv[i+1]

        # Define spam executable location
        elif arg == '-uID':
            try:
                uniqID = int(argv[i+1])
            except:
                print('\'%s\' is an invalid number spam basic run unique id' %  argv[i+1] )
                print('Exiting...\n')
                exit(-1)
        # end '-n'


        # Specify the number of particles in each galaxy
        elif arg == '-n':
            try:
                nPart = int(argv[i+1])
            except:
                print('\'%s\' is an invalid number for the number of particles' %  argv[i+1] )
                print('Exiting...\n')
                exit(-1)
        # end '-n'
 
        # Output Directory
        elif arg == '-o':
            outputDir = argv[i+1]
            if outputDir[-1] != '/':
                outputDir = outputDir + '/'
       
       
        # Check if wants a run directory or not
        elif arg == '-overwrite':
            overWrite = True

        # Check if wants a run directory or not
        elif arg == '-createDir':
            makeRunDir = True

        elif arg == '-noDir':
            makeRunDir = False

        elif arg == '-compress':
            compressFiles = True

        # Specify SDSS Name
        elif arg == '-sdss':
            sdssName = argv[i+1]

        # Specify run numer
        elif arg == '-run':
            try:
                runNum = int(argv[i+1])
            except:
                print('\'%s\' is invalid for run number' %  argv[i+1] )
                print('Exiting...\n')
                exit(-1)

        elif arg == '-gen':
            try:
                genNum = int(argv[i+1])
            except:
                print('\'%s\' is invalid generation number' %  argv[i+1] )
                print('Exiting...\n')
                exit(-1)

        # Just one model with data via command line
        elif arg == '-m':
            useModelData = True
            modelData = argv[i+1]


        # Define Zoo Model File
        elif arg == '-zoofile':
            zooFile = argv[i+1]
            useZooFile = True

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

    # end looping through arguments


    # This double checks if inputs are meaninful

    exitProgram = False

    # Is number of particle practical? 
    if nPart <= 0:
        print('Number of particles must be greater than 0')
        exitProgram = True

    elif nPart > maxN:
        print('Not accepting number of particle greater than %d' % maxN)
        exitProgram = True

    # Is basic run an actual file? 
    isFile = path.isfile(basicRun)
    if not isFile:
        print('SPAM Basic Run executable at \'%s\' was not found' % basicRun)
        exitProgram = True


    if useZooFile and useModelData:
        print('Both zoo file and model data given.')
        print('Please choose one or the other.')
        exitProgram = True

    if not useZooFile and not useModelData:
        print('Neither zoo file nor model data given.')
        print('Please choose one or the other.')
        exitProgram = True

    if useZooFile: 

        if nZooRuns == 0 and allZooRuns == False:
            print('Please specify how many runs you would like from zoo file')
            exitProgram = True

        if nZooRuns > 0 and allZooRuns == True:
            print('Please specify either all Zoo runs or a Number of zoo runs')
            exitProgram = True

        isFile = path.isfile(zooFile)
        if not isFile:
            print('Galaxy Zoo model file \'%s\' not found' % zooFile)
            print('Please specify by adding \'-zooFile zooDirectory/zooFile.txt\' as command line argument')
            exitProgram = True

    if useModelData:
        lData = len(modelData.split(','))
        if lData != 34:
            print( 'Model Data was not the correct format.' )
            print( 'Expected format is 34 floating point numbers seperated by commas, no spaces')
            exitProgram = True

    # Exit program with error
    if exitProgram:
        print('Exiting...\n')
        exit(-1)

# End readCommandLine()


# Execute everything after declaring functions
main()
