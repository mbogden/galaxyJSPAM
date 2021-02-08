'''
    Author:     Matthew Ogden
    Created:    20 July 2019
    Altered:    21 July 2019
Description:    This program is to make executable lines for pixel_comparison.py
'''

from sys import \
        exit, \
        argv, \
        stdout

from os import \
        path, \
        listdir


# For this program
printAll = True

# For cmds
cmdPrint = False
argFileLoc = ''

sdssDir = ''
sdssZooFile = ''

nLines = 0
writeLoc = ''

methodName = ''

def main():

    endEarly = readArg()

    if printAll:
        print('sdssDir: %s' % sdssDir)

    if endEarly:
        print('Exiting cmd_compare_maker.py...')
        exit(-1)

    runDirList = listdir( sdssDir )
    runDirList.sort()

    #print(runDirList)

    oFile = open( writeLoc, 'w' )
    for i,runDir in enumerate(runDirList):

        # Check if directory exists
        if not path.exists( sdssDir + runDir ):
            continue

        # check if its a run directory
        if 'run' not in runDir:
            continue


        cmd = 'python3'
        cmd += ' Comparison_Methods/compare_v1.py'
        cmd += ' -runDir %s' % sdssDir + runDir + '/'
        cmd += ' -argFile %s' % argFileLoc

        if not cmdPrint:
            cmd += ' -noprint'

        cmd += '\n'

        oFile.write(cmd)

        if i < 2:
            print(cmd)

        if not nLines == 0 and i >= nLines:
            break

# End main

def readArg():

    global printAll, sdssDir, writeLoc, sdssZooFile, nLines, cmdPrint, argFileLoc
    endEarly = False

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            #argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-cmdPrint':
            cmdPrint = True

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]

        elif arg == '-writeLoc':
            writeLoc = argList[i+1]

        elif arg == '-zooFile':
            sdssZooFile = argList[i+1]

        elif arg == '-n':
            nLines = int( argList[i+1] )

        elif arg == '-methodName':
            methodName = argList[i+1]



    # Check if input arguments were valid

    if writeLoc == '':
        print('Please specify file you would like to store executable lines')
        endEarly = True

    if not path.exists(sdssDir):
        print('sdss directory %s not found' % sdssDir)
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
        # End going through file
# end read argument file

# Run main after declaring functions
main()
