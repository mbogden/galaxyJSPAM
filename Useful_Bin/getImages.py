'''
    Author:     Matthew Ogden
    Created:    20 July 2019
    Altered:    19 Sep 2019
Description:    This program is meant to get training images for a filter
'''

from sys import \
        exit, \
        argv, \
        stdout

from os import \
        path, \
        listdir, \
        system

import pandas as pd

from re import findall

printAll = True

sdssDir = ''
mainDir = ''

nGood = 50
nBad = 50

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
        print('nGoodImg: %d' % nGood)
        print('nBadImg : %d' % nBad)

    if endEarly:
        print('Exiting...')
        exit(-1)

    #Create image directories if they don't exist
    if not path.exists(goodDir):
        print("Creating dir: %s" % goodDir)
        system("mkdir -p %s" % goodDir)

    if not path.exists(badDir):
        print("Creating dir: %s" % badDir)
        system("mkdir -p %s" % badDir)

    
    print("Going through dirs:")
    if mainDir != '':
        allSdssDirs = listdir(mainDir)
        for sDir in allSdssDirs:
            sdssDir = mainDir + sDir + '/'
            print('\t',sdssDir)
            sdss_dir(sdssDir)

# End main

def sdss_dir( sdssDir ):

    runDirList = listdir( sdssDir )
    parsedSdss = sdssDir.split('/')
    sdssName = parsedSdss[5]

    toFrameList = []

    # get run # from dir name
    for rDir in runDirList:

        if 'run' not in rDir:
            continue
        
        parsedRun = rDir.split('_')

        if len(parsedRun) != 3:
            print("ERROR.  Found: %s" % rDir)
            continue

        toFrameList.append( [ sdssDir + rDir , int(parsedRun[2]) ] )
        
    # End looping through runs for #'s
    print("\t\t Found %d runs" % len(toFrameList))

    runFrame = pd.DataFrame( toFrameList, columns = ['runPath', 'runNum'] )

    # Loop through top n runs for good images
    runFrame = runFrame.sort_values( by=['runNum'] ).reset_index()
    for i,row in runFrame.head(n=nGood).iterrows():
        rPath = row.runPath + '/'
        rNum = row.runNum
        copyImg( sdssName, rNum, rPath, goodDir)

    # Reverse and go through bottom n images 
    runFrame = runFrame.sort_values(ascending=False, by=['runNum'] ).reset_index()
    for i,row in runFrame.head(n=nGood).iterrows():
        rPath = row.runPath + '/'
        rNum = row.runNum
        copyImg( sdssName, rNum, rPath, badDir)


# End sdss_dir

def copyImg( sdssName, rNum, rPath, toDir):

    # Find images
    rFiles = listdir(rPath)
    
    modelPath = ''
    initPath = ''

    for rFile in rFiles:

        if 'model.png' in rFile:
            modelPath = rPath + rFile

        if 'model_init.png' in rFile:
            initPath = rPath + rFile

    # Make sure you found both images
    if modelPath == '' or initPath == '':
        print("Error: Did not find images in: %s" % rPath)
        return

    toModelPath = toDir + '%s_%d_model.png' % (sdssName, rNum )
    toInitPath = toDir + '%s_%d_init.png' % ( sdssName, rNum )
    
    system("cp %s %s" % ( modelPath, toModelPath ))
    system("cp %s %s" % ( initPath, toInitPath ))


# end copy Img
 
def readArg():

    global printAll, mainDir, imgDir, goodDir, badDir
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
            if imgDir[-1] != '/':
                imgDir += '/'
            goodDir = imgDir + goodDir
            badDir  = imgDir + badDir

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
