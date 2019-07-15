'''
    Author:     Matthew Ogden
    Created:    12 Jul. 2019
    Altered:    12 Jul. 2019
Description:    This python script is intended for executing many machine comparisons
                    of images located within directories or run folders.
                    NOTE: at this time, it is only programed to be pointed at a sdssDirectory.
'''

import os
import sys

# Global input arguments
execLoc = ''

pyExecLoc = ''
usePython = ''

addFlag = ''

sdssDir = ''
genNum = ''
paramName = ''

targetLoc = ''
targetInfoLoc = ''
targetCenters = ''


def main():
    print('In batch_comp.py')

    endEarly = readArg()

    if endEarly:
        print('Exiting...')
        sys.exit(-1)

    printInfo()

    traverseAndExecute()

#end Main

def traverseAndExecute():

    # get run directories.
    dirList = os.listdir(sdssDir)
    runList = []
    for f in dirList:
        if 'run' in f:
            runList.append(f)

    # Filter gen number as needed
    if genNum != '':
        newRunList = []

        # loop through run directories checking their gen number
        for run in runList:
            gNum = run.split('_')[1]
            if gNum == genNum:
                newRunList.append(run)

        print('Found %d of %d runs with genNum \'%s\'.' % ( len(newRunList), len(runList), genNum ))
        runList = newRunList
    # end filtering by gen number

    
    # Loop through run directories
    for i,run in enumerate(runList):
        runDir = sdssDir + run + '/'

        # Get files in runDir
        runFiles = os.listdir(runDir)

        imageLoc = ''
        infoLoc = ''
        scoreLoc = ''
        

        # Loop through files identifying them
        for f in runFiles:

            if '.png' in f and paramName in f:
                imageLoc = runDir + f

            elif 'info.txt' == f:
                infoLoc = runDir + f

        # Check if any files are missing
        if imageLoc == '':
            print('Could not find any model images in \'%s\' with parameter name \'%s\'' % ( runDir, paramName) )
            continue

        if infoLoc == '':
            print('Info file in \'%s\' not found' % runDir )
            continue

        scoreLoc = runDir + 'scores.csv'

        # Assuming both nessesary files are there.  execute

        execute( imageLoc, infoLoc, scoreLoc )

        if i == 2:
            break

#end traverse and execute 

def execute(imageLoc, infoLoc, scoreLoc):

    # build executable command piece by piece

    cmd = ''

    if pyExecLoc != '':
        cmd = 'python3 %s' % pyExecLoc
    elif execLoc != '':
        cmd = './%s' % execLoc
    else:
        print('Warning... You should not be seeing this.  skipping.')
        return -1

    cmd = cmd + ' -image %s'    % imageLoc
    cmd = cmd + ' -imgInfo %s'  % infoLoc
    cmd = cmd + ' -target %s'   % targetLoc

    if targetInfoLoc != '':
        cmd = cmd + ' -targetInfo %s'   % targetInfoLoc
    elif targetCenter != '':
        cmd = cmd + ' -targetCenter %s' % targetCenters
    else:
        print('Warning... You should not be seeing this. skipping.')
        return -1

    cmd = cmd + ' -score %s'    % scoreLoc
    
    #print('About to execute \n\t$: %s\n' % cmd )

    os.system(cmd)

#end execute



def printInfo():

    if execLoc != '':
        print('Using executable \'%s\' for comparison method' % execLoc)
    elif pyExecLoc != '':
        print('Using python executable \'%s\' for comparison method' % pyExecLoc)
    else:
        print('\nYou shouldn\'t be seeing this.  executable locations blank')
        sys.exit(-1)

    print('Using run images found within directory \'%s\'' % sdssDir)
    print('Using target image \'%s\'' % targetLoc )

    if targetInfoLoc != '':
        print('Using target info file \'%s\'' % targetInfoLoc)
    elif targetCenters != '':
        print('Using target centers \'%s\'' % targetCenters)
    else:
        print('\nYou shouldn\'t be seeing this.  Target info blank')
        sys.exit(-1)

    if genNum != '':
        print('Using only generation \'%s\' images.' % genNum)

    if paramName != '':
        print('Using only param \'%s\' images.' % paramName)


# end print info



def readArg():

    global execLoc, pyExecLoc, sdssDir, genNum, paramName
    global targetLoc, targetInfoLoc, targetCenters, addFlag

    endEarly = False

    for i,arg in enumerate(sys.argv):

        # ignore argument until specifier is found
        if arg[0] != '-':
            continue

        elif arg == '-pyExec':
            pyExecLoc = sys.argv[i+1]
            usePython = True

        # Future implementation.  For using generic linux executable files
        elif arg == '-exec':
            execLoc = sys.argv[i+1]
            print('Warning!  Using normal executables not available at this time.')
            endEarly = True

        elif arg == '-targetImg':
            targetLoc = sys.argv[i+1]

        elif arg == '-targetInfo':
            targetInfoLoc = sys.argv[i+1]

        # Expecting -targetCenter 100,100,400,400 format
        elif arg == '-targetCenter':
            targetCenters = sys.argv[i+1]


        elif arg == '-sdssDir':
            sdssDir = sys.argv[i+1]
            if sdssDir[-1] != '/':
                sdssDir = sdssDir + '/'

        # future implementation
        elif arg == '-param':
            paramName = sys.argv[i+1]

        # Future implementation.  Only go through specified generation
        elif arg == '-gen':
            genNum = sys.argv[i+1]

        # Current implementation confusing.
        # Meant for me to specify what hard coded additional arguments I want to pass to executable
        # Could not think of a quick convenient way to pass additional arguments as string in command line
        elif arg == '-flag':
            addFlag = sys.argv[i+1]

    # end looping through arguments

    # Check if irguments are valid

    # Check if given python exec exists
    if not os.path.isfile(pyExecLoc):
        print('No executable file given for comparison')
        endEarly = True


    # check if sdss directory given exists
    if not os.path.exists(sdssDir):
        print( 'Directory \'%s\' not found')
        endEarly = True

    # check is sdss directory has run folders
    dirList = os.listdir( sdssDir )

    foundRun = False
    for f in dirList:
        if 'run' in f:
            foundRun = True
            break

    if not foundRun:
        print('No run directories found within \'%s\'' % sdssDir)
        endEarly = True


    # Current implementation requires specifiying a param for image
    if paramName == '':
        print('Current implementation requires specifying a parameter name for image')
        endEarly = True

    # Check if target image exists
    if not os.path.isfile(targetLoc):
        print( 'Target image at \'%s\' not found' % targetLoc )
        endEarly = True

    # Chech if information target image is given
    if not os.path.isfile( targetInfoLoc ) and targetCenters == '':
        if not os.path.isfile( targetInfoLoc ):
            print( 'Target info file at \'%s\' not found' % targetInfoLoc )
        elif targetCenters == '':
            print( 'No galaxy centers given for target image')
        endEarly = True

    return endEarly
# End read Arg


main()
