'''
    Author:     Matthew Ogden
    Created:    20 July 2019
    Altered:    17 Sep 2019
Description:    This program is to make executable lines for Image Creator v3
'''

from sys import \
        exit, \
        argv, \
        stdout

from os import \
        path, \
        listdir


printAll = True
noPrint = False

sdssDir = ''
mainDir = ''

nLines = 0
writeLoc = ''
paramLoc = ''

def main():

    endEarly = readArg()

    if printAll:
        print('mainDir: %s' % mainDir)
        print('sdssDir: %s' % sdssDir)
        print('Write Loc: %s' % writeLoc)
        print('Print %d lines' % nLines)

    if endEarly:
        print('Exiting cmd_compare_maker.py...')
        exit(-1)


    oFile = open( writeLoc, 'w' )

    i = 0
    if mainDir != '':
        sdssDirs = listdir(mainDir)
        for sDir in sdssDirs:
            sDir = mainDir + sDir + '/'
            print(sDir)
            hitMax = cmd_sdss_dir( sDir, oFile, i )
            if hitMax:
                break
            

# End main

def cmd_sdss_dir( sdssDir, oFile, i ):

    runDirList = listdir( sdssDir )
    runDirList.sort()

    for runDir in runDirList:

        if not path.exists( sdssDir + runDir ):
            continue

        if 'run' not in runDir:
            continue

        if not nLines == 0 and i >= nLines:
            return True
        i += 1

        cmd = 'python3'
        cmd += ' Image_Creator/python/image_creator_v3.py'
        cmd += ' -runDir %s' % sdssDir + runDir + '/'
        cmd += ' -paramLoc %s' % paramLoc
        cmd += ' -initial'
        cmd += ' -dotImg'

        if noPrint:
            cmd += ' -noprint'

        cmd += '\n'

        oFile.write(cmd)


    return False

# End cmd_sdss_dir

 
def readArg():

    global printAll, sdssDir, writeLoc, sdssZooFile, nLines, paramLoc, noPrint, mainDir
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

        elif arg == '-mainDir':
            mainDir = argList[i+1]

        elif arg == '-writeLoc':
            writeLoc = argList[i+1]

        elif arg == '-paramLoc':
            paramLoc = argList[i+1]

        elif arg == '-n':
            nLines = int( argList[i+1] )

        elif arg == '-noprint':
            noPrint = True


    # Check if input arguments were valid

    if writeLoc == '':
        print('Please specify file you would like to store executable lines')
        endEarly = True

    if not path.exists(sdssDir) and not path.exists(mainDir):
        if not path.exists(sdssDir):
            print('Directories not found:')
            print('\tsdss:%s' % sdssDir)
            print('\tmain:%s' % mainDir)
        endEarly = True

    if not path.exists(paramLoc):
        if paramLoc == '':
            print("Please provide Image Parameter file")
        else:
            print("Image Parameter file does not exist:\n\t%s" % paramLoc)
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

            # Skip line if empty
            if len(l) == 0:
                continue

            # Skip line if comment
            if l[0] == '#':
                continue

            lineItems = l.split()
            for item in lineItems:
                argList.append(item)

        # End going through file

# end read argument file

# Run main after declaring functions
main()
