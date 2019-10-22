'''
     Author:    Matthew Ogden
    Created:    21 Oct 2019
    Altered:    
Description:    Created for simple comparison of sdss images
'''

from sys import \
        exit, \
        argv

from os import \
        listdir, path

import numpy as np
import cv2
import matplotlib.pyplot as plt

printAll = True

# global input variables
imgDir = ''
everyN = 1

def main():

    endEarly = readArg()

    if printAll: print("ImgDir: %s" % imgDir )

    if endEarly: exit(-1)

    tImg = cv2.imread( imgDir + 'target.png' )
    if True:
        recreateData(tImg)

    # read all files
    hScores = readScores( 'scoreFile.txt' )
    pScores = readScores( 'pScores.txt' )
    mScores = readScores( 'pixelDiffScores.txt' )

    print(len(hScores))
    print(len(pScores))
    print(len(mScores))


    createHeatPlot( hScores, mScores, pScores, 'plot_t.png' )

# End main

def createHeatPlot( hScores, mScores, pScores, saveLoc ):


    pMin = np.amin( pScores )
    pMax = np.amax( pScores )

    cMap = plt.cm.get_cmap('RdYlBu_r')
    plot = plt.scatter( hScores, mScores, c=pScores, vmin = pMin, vmax = pMax, s=10, cmap=cMap)
    cBar = plt.colorbar(plot)

    plt.title("Human vs Machine Score with Perturbedness")
    plt.xlabel("Human Scores")
    plt.ylabel("Pixel Differenece from Target")

    cBar.set_label("Pixel difference from unperturbed self")

    plt.savefig( saveLoc )


def recreateData(tImg):

    mImgs, iImgs = getImgs()

    pScores = createPScores( mImgs, iImgs )
    mScores = createMScores( mImgs, tImg )

    saveScores( pScores, 'pScores.txt' )
    saveScores( mScores, 'pixelDiffScores.txt' )



def readScores( fileName ):

    inFile = readFile( imgDir + fileName )

    nLines = len( inFile )

    scores = np.zeros( nLines )

    for i, val in enumerate(inFile):
        val = float(val.strip())
        scores[i] = val

    if everyN == 1:
        return scores

    else:

        nScores = []

        for i, val in enumerate( scores ):

            if i % everyN == 0:
                nScores.append( val )

        return np.array( nScores )


def saveScores( scores, fileName ):
    
    oFile = open( imgDir +  fileName , 'w' )

    for val in scores:
        oFile.write( "%f\n" % val )

    oFile.close()


def createMScores( mImgs, tImg ):

    mScores = np.zeros( len( mImgs ))

    tB = np.sum( tImg )

    for i,mImg in enumerate(mImgs):

        # adjust so mean brightness matches
        mB = np.sum( mImg )
        mImg = mImg * ( tB / mB )

        dImg = np.abs( mImg - tImg )
        mVal = np.sum( dImg ) / dImg.size / 255
        mVal = 1 - mVal

        # arbitrary adjustment to make score more human readable
        mVal = 10* (mVal - 0.9)

        mScores[i] = mVal

    return mScores

# end createMScores()


def createPScores( mImgs, iImgs ):

    pScores = np.zeros( len( mImgs ))

    for i in range( len( mImgs )):

        dImg = np.abs( mImgs[i] - iImgs[i] )
        pVal = np.sum( dImg ) / dImg.size / 255
        pScores[i] = pVal

    return pScores

# end createPScores()

def getImgs():
    mImgs = []
    iImgs = []

    allFiles = listdir( imgDir )
    allFiles.sort()

    for f in allFiles:
        try:
            rNum = int( f.split('_')[0] )
        except:
            continue
        
        if rNum % everyN == 0:

            f = imgDir + f

            if 'model' in f:
                mImgs.append( cv2.imread( f ) )

            if 'init' in f:
                iImgs.append( cv2.imread( f ) )

    if len(mImgs) == 0 or len(iImgs) == 0:
        print("# of model and init images 0")
        exit(0)

    if len(mImgs) != len(iImgs):
        print("# of model and init images different")
        exit(0)

    return mImgs, iImgs

# End getImgs

def getHumanScores():

    allFiles = listdir( imgDir )
    nRuns = int( ( len(allFiles) - 2 ) / 3 )
    hScores = np.zeros(nRuns)

    sFile = readFile( imgDir + 'scoreFile.txt' )

    for i, score in enumerate(sFile):
        hScores[i] = float(score.strip())

    if everyN == 1:
        return hScores
    else:

        nhScores = []
        for i,s in enumerate(hScores):
            if i % everyN == 0:
                nhScores.append(s)
        hScores = np.array( nhScores )
        return hScores

    '''

    for f in allFiles:

        # Skip if no human score
        if not 'info' in f: continue

        iFile = readFile( imgDir + f )
        rNum = -1
        hScore = -1.0

        for l in iFile:
            l = l.strip()

            if 'run_number' in l:
                rNum = int(l.split()[1])

            if 'human_score' in l:
                hScore = float(l.split()[1])

        if rNum == -1 or hScore == -1.0:
            print("ERROR")
            continue

        hScores[rNum] = hScore

    for i, s in enumerate(hScores):
        sFile.write("%f\n" % s )

    '''

# end getHumanScores


def readArg():

    global printAll, imgDir, everyN

    argList = argv
    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-imgDir':
            imgDir = argList[i+1]

        elif arg == '-n':
            everyN = int( argList[i+1] )

    # Check if input arguments were valid
    if imgDir == '':
        print("Please specify image directory")
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
    return argList
# end read argument file

def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return []
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return []

    else:
        inList = list(inFile)
        inFile.close()
        return inList

# End simple read file

# Run main after declaring functions
main()
