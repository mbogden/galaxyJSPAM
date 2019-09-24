'''
Author:     Matthew Ogden
Created:    10 May 2019
Altered:    24 Sep 2019
Purpose:    This program is the main function that prepares galaxy models, 
                runs them through the JSPAM software, and creates particle files
'''

from sys import (
        argv,
        exit
        )

from os import (
        path,
        system
        )

# Global/Input Variables

printAll = True

basicRun = 'Simulator/bin/basic_run'
uniqID = 0

runDir = ''
nPart = 0
maxN = 1e6      # Arbitrary limit in case of typo, can be changed if needed

compressFiles = True
overWrite = False


# SE_TODO: Work with this function for main
def new_main():

    endEarly = new_readArg()

    # SE: These should all have a value
    if printAll:
        print("Run Dir : %s" % runDir)
        print("Num Part: %d" % nPart)

    # Exit program with error
    if endEarly:
        print('Exiting...\n')
        exit(-1)

    # SE_TODO: Create function to read info.txt located in runDir
    '''
    1. Check if info file exists. exit with print statement if not
    2. Open file, get info for model_data (after space), and return as string 
    '''

    # SE_TODO: Create 'particle_files' folder in runDir if it does not exist

    # SE_TODO: Check 'particle_files' folder if particles of nPart size have already been made 
    # Ignore if overWrite is true

    # SE_TODO: Move active directory into 'particle_files'. 

    # SE_TODO: Call JSPAM! 
    # runBasicRun( nPart, mData )

    # SE_TODO: Check if particle files were created.  Should end with .000 and .101
    # SE_TODO: Rename both files as nPart_pts.000 and nPart_pts.101 (Ex. 1000_pts.000)
    # SE_TODO: Zip both files up in 1 zip file named nPart_pts.zip
    # SE_TODO: Delete .000 and .101 if they remain




def old_main():

    endEarly = readArg()

    # Exit program with error
    if endEarly:
        print('Exiting...\n')
        exit(-1)

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


# End main


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
            if printAll:
                print('Zipping files')

            zipCmd1 = 'zip a_%d_000.zip a_%d.000' % (uID, uID)
            zipCmd2 = 'zip a_%d_101.zip a_%d.101' % (uID, uID)

            system(zipCmd1)
            system(zipCmd2)

            if printAll:
                print('Moving files to %s' % oDir)

            if pName == '':
                mvCmd = 'mv a_%d_*.zip %s' % (uID, oDir)
                system(mvCmd)
            else:
                mv1Cmd = 'mv a_%d_000.zip %s' % (uID, oDir + pName + '_000.zip')
                mv2Cmd = 'mv a_%d_101.zip %s' % (uID, oDir + pName + '_101.zip')
                system(mv1Cmd)
                system(mv2Cmd)


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



def runBasicRun( nPart, data ):

    global basicRun

    if basicRun[0] != '/':
        basicRun = '/' + basicRun
        
    sysCmd = '.%s -m %d -n1 %d -n2 %d %s' % ( basicRun, 0, nPart, nPart, data )

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


def new_readArg():

    # Global input arguments
    global runDir, nPart, overWrite

    # Get list of arguments from command line
    argList = argv

    # Loop through command line arguments
    for i,arg in enumerate(argList):

        # Ignore argument unless it has '-' specifier
        if arg[0] != '-':
            continue
        
        elif arg == '-runDir':
            runDir = argList[i+1]

        elif arg == '-nPart':
            nPart = argList[i+1]

        elif arg == '-overWrite':
            overWrite = True



    # Check validity of commandline arguments
    endEarly = False

    # Check if run Dir exists or if not given one
    if not path.exists(runDir):
        
        if runDir == '':
            print("ERROR: Please specify path to run directory.")
            print("\t$: python simulator.py -runDir path/to/runDir") 
        else:
            print("ERROR: Path to run directory not found: '%s'" % runDir)
        endEarly = True
    # End check runDir


    # Check if number of particles was specified
    if nPart == 0:
        print("ERROR: Please specify number of particles per galaxy.")
        print("\t$: python simulator.py -nPart 1000")
        endEarly = True
    
    # Check if input particles was an integer
    try:
        nPart = int(nPart)
    except:
        print("ERROR: -nPart is not an integer: '%s'" % nPart)
        nPart = 0
        endEarly = True


    return endEarly







def readArg():

    global basicRun, nPart, outputDir, makeRunDir
    global sdssName, runNum, genNum, uniqID, compressFiles, pName
    global useZooFile, zooDir, zooFile, allZooRuns, nZooRuns
    global useModelData, modelData, modelName, humanScore, printAll, overWrite

    argList = argv

    for i,arg in enumerate(argList):

        if arg == '-0':
            print('You found Zero')

        # Define if you want printing
        elif arg == '-print':
            printAll = True

        # Define spam executable location
        elif arg == '-spam':
            basicRun = argList[i+1]

        # Define spam executable location
        elif arg == '-uID':
            try:
                uniqID = int(argList[i+1])
            except:
                print('\'%s\' is an invalid number spam basic run unique id' %  argList[i+1] )
                print('Exiting...\n')
                exit(-1)
        # end '-n'


        # Specify the number of particles in each galaxy
        elif arg == '-n':
            try:
                nPart = int(argList[i+1])
            except:
                print('\'%s\' is an invalid number for the number of particles' %  argList[i+1] )
                print('Exiting...\n')
                exit(-1)
        # end '-n'
 
        # Output Directory
        elif arg == '-o':
            outputDir = argList[i+1]
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


        # Specify name for particle file
        elif arg == '-name':
            pName = argList[i+1]

        # Specify SDSS Name
        elif arg == '-sdss':
            sdssName = argList[i+1]

        # Specify run numer
        elif arg == '-run':
            try:
                runNum = int(argList[i+1])
            except:
                print('\'%s\' is invalid for run number' %  argList[i+1] )
                print('Exiting...\n')
                exit(-1)

        elif arg == '-gen':
            try:
                genNum = int(argList[i+1])
            except:
                print('\'%s\' is invalid generation number' %  argList[i+1] )
                print('Exiting...\n')
                exit(-1)

        # Just one model with data via command line
        elif arg == '-modelData':
            useModelData = True
            modelData = argList[i+1]

        # Just one model with data via command line
        elif arg == '-modelName':
            useModelData = True
            modelName = argList[i+1]

        # Just one model with data via command line
        elif arg == '-humanScore':
            humanScore = argList[i+1]


        # Define Zoo Model File
        elif arg == '-zoofile':
            zooFile = argList[i+1]
            useZooFile = True

        # Do N number of runs in zoo model file
        elif arg == '-zn':
            try:
                nZooRuns = int(argList[i+1])
            except:
                print('\'%s\' is an invalid number for the number zoo models' %  argList[i+1] )
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

    return exitProgram

# End read argumnets()


# Execute everything after declaring functions
print('')
new_main()
print('')
