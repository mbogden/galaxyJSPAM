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
allDir = 'allImgs/'


def main():


    endEarly = readArg()

    if printAll:
        print('mainDir : %s' % mainDir)
        print('sdssDir : %s' % mainDir)
        print('imgDir  : %s' % imgDir)
        print('allDir  : %s' % imgDir)
        print('goodDir : %s' % goodDir)
        print('badDir  : %s' % badDir)
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

    if not path.exists(allDir):
        print("Creating dir: %s" % allDir)
        system("mkdir -p %s" % allDir)

    getSdssInitImages(sdssDir)
    
    return 0
'''
    print("Going through dirs:")
    if mainDir != '':
        allSdssDirs = listdir(mainDir)
        for sDir in allSdssDirs:
            sdssDir = mainDir + sDir + '/'
            print('\t',sdssDir)
            sdss_dir(sdssDir)
'''

# End main


def getSdssInitImages( sdssDir ):
    print("sdssDir: %s" %  sdssDir)

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

    runFrame = runFrame.sort_values( by=['runNum'] ).reset_index()
    for i,row in runFrame.iterrows():
        rPath = row.runPath + '/'
        rNum = row.runNum
        copyInitImg( sdssName, rNum, rPath, imgDir)

def copyInitImg( sdssName, rNum, rPath, toDir):

    # Find images
    rFiles = listdir(rPath)
    
    imgPath = ''

    for rFile in rFiles:

        if 'init.png' in rFile:
            imgPath = rPath + rFile

    if imgPath == '':
        print("Error: Did not find images in: %s" % rPath)
        return

    toImgPath = toDir + '%d_init.png' % (rNum )
    
    print("cp %s %s" % ( imgPath, toImgPath ))
    system("cp %s %s" % ( imgPath, toImgPath ))





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

    runFrame = runFrame.sort_values( by=['runNum'] ).reset_index()
    for i,row in runFrame.iterrows():
        rPath = row.runPath + '/'
        rNum = row.runNum
        copyImg( sdssName, rNum, rPath, imgDir)


    '''
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
    '''

# End sdss_dir

def copyImg( sdssName, rNum, rPath, toDir):

    # Find images
    rFiles = listdir(rPath)
    
    diffPath = ''

    for rFile in rFiles:

        if 'model.png' in rFile:
            diffPath = rPath + rFile

    # Make sure you found both images
    if diffPath == '':
        print("Error: Did not find images in: %s" % rPath)
        return

    toDiffPath = toDir + '%s_%d_model.png' % (sdssName, rNum )
    
    print("cp %s %s" % ( diffPath, toDiffPath ))
    system("cp %s %s" % ( diffPath, toDiffPath ))


# end copy Img
 
def readArg():

    global printAll, mainDir, imgDir, goodDir, badDir, allDir, sdssDir
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

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]

        elif arg == '-imgDir':
            imgDir = argList[i+1]
            print(imgDir)
            if imgDir[-1] != '/':
                imgDir += '/'

            goodDir = imgDir + goodDir
            badDir  = imgDir + badDir
            allDir  = imgDir + allDir

        elif arg == '-n':
            nLines = int( argList[i+1] )

        elif arg == '-noprint':
            noPrint = True

    # Check if input arguments were valid

    endEarly = False

    if imgDir == '':
        print('ERROR: \tPlease enter Image Directory')
        endEarly = True

    if not path.exists(mainDir) and not path.exists(sdssDir):
        print('ERROR: \tmainDir not found: \'%s\'' % mainDir)
        print('ERROR: \tsdssDir not found: \'%s\'' % mainDir)

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
