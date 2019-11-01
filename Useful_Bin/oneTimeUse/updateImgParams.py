'''
    Author:     Matthew Ogden
    Created:    20 July 2019
    Altered:    19 Sep 2019
Description:    This program is temporary get images
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
mainDir = ''

nGoodImg = 2
nBadImg = 2

imgDir = ''
goodDir = 'goodImgs/'
badDir = 'badImgs/'


def main():

    endEarly = readArg()

    if printAll:
        print('mainDir : %s' % mainDir)
        print('imgDir  : %s' % imgDir)
        print('goodDir : %s' % imgDir + goodDir)
        print('badDir  : %s' % imgDir + badDir)
        print('nGoodImg: %d' % nGoodImg)
        print('nBadImg : %d' % nBadImg)

    if endEarly:
        print('Exiting...')
        exit(-1)


    i = 0
    print("Going through dirs:")
    if mainDir != '':
        sdssDirs = listdir(mainDir)
        for sDir in sdssDirs:
            sDir = mainDir + sDir + '/'
            print('\t',sDir)

# End main

def sdss_dir( sdssDir ):

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

        oFile.write(cmd)

    return False

# End cmd_sdss_dir

 
def readArg():

    global printAll, mainDir, imgDir
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

        elif arg == '-mainDir':
            mainDir = argList[i+1]

        elif arg == '-imgDir':
            imgDir = argList[i+1]

        elif arg == '-n':
            nLines = int( argList[i+1] )

        elif arg == '-noprint':
            noPrint = True

    # Check if input arguments were valid

    if imgDir == '':
        print('Warning: Please enter Image Directory')
        endEarly = True

    if not path.exists(mainDir):
        print('\tmainDir not found: \'%s\'' % mainDir)
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
