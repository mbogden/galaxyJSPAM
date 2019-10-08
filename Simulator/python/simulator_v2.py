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
        system,
        getcwd
        )

# Global/Input Variables

printAll = True

basicRun = 'Simulator/bin/basic_run'
if path.exists(basicRun):
    basicRun = path.abspath(basicRun)
else:
    print("Can't find basic_run. Exiting....")
    print("basic_run location: %s" % basicRun)
    print("current location: %s" % getcwd())
    exit('-1')

runDir = ''
nPart = 0
maxN = 1e6      # Arbitrary limit in case of typo, can be changed if needed

compressFiles = True
overWrite = False


# SE_TODO: Work with this function for main
def simulator_v2( argList ):

    endEarly = new_readArg( argList )

    # SE: These should all have a value
    if printAll:
        print("Run Dir : %s" % runDir)


    # Exit program with error
    if endEarly:
        print('Exiting...\n')
        exit(-1)

    # SE_TODO: Create function to read info.txt and get model data located in runDir

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


def new_readArg(argList):

    # Global input arguments
    global runDir, nPart, overWrite

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
# End new_readArg







# Execute everything after declaring functions
print('')
if __name__=="__main__":
    argList = argv
    simulator_v2( argList )
print('')
