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


printAll = True

sdssDir = ''
sdssZooFile = ''

nLines = 0
writeLoc = ''

def main():

    endEarly = readArg()

    if endEarly:
        print('Exiting cmd_compare_maker.py...')
        exit(-1)

    runDirList = listdir( sdssDir )

    zooFile = open( sdssZooFile, 'r')

    #print(runDirList)

    oFile = open( writeLoc, 'w' )
    for i,runDir in enumerate(runDirList):

        if not path.exists( sdssDir + runDir ):
            continue

        if 'run' not in runDir:
            continue

        l = zooFile.readline().strip()

        temp1, zooModelStr = l.split('\t')

        zooModelName, humanScore, temp, temp = temp1.split(',')

        cmd = 'python3'
        cmd += ' Comparison_Methods/compare.py'
        cmd += ' -runDir %s' % sdssDir + runDir + '/'
        cmd += ' -argFile Input_Data/comparison_methods/arg_compare.txt'
        cmd += ' -humanScore %s' % humanScore
        cmd += ' -zooModel %s' % ( '%s:%s' % ( zooModelName, zooModelStr ) )

        cmd += '\n'

        oFile.write(cmd)

        if i < 2:
            print(cmd)

        if not nLines == 0 and i > nLines:
            break

# End main

def readArg():

    global printAll, sdssDir, writeLoc, sdssZooFile, nLines
    endEarly = False

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]

        elif arg == '-writeLoc':
            writeLoc = argList[i+1]

        elif arg == '-zooFile':
            sdssZooFile = argList[i+1]

        elif arg == '-n':
            nLines = int( argList[i+1] )



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
