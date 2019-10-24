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

imgScale = 0.1

def main():

    endEarly = readArg()

    if printAll: print("ImgDir: %s" % imgDir )

    if endEarly: exit(-1)

    tImg = cv2.imread( imgDir + 'target.png' )

    '''
    tImgs = []
    nImg = np.zeros( tImg.shape)

    bLines = np.linspace( 0, 200, 7)

    for i,threshold in enumerate(bLines):
        tImgs.append( binImg( tImg, threshold ) )
        #cv2.imwrite( 'target_binary_%d.png' % i, tImgs[-1] )

        nImg += tImgs[-1]


    bMax = np.amax( nImg )

    nImg = nImg * 255 / bMax

    #tImg = binImg( tImg, 45 )

    cv2.imwrite( 'target_3.png', nImg )

    return
    '''

    if False:
        recreateData(tImg)

    if False:
        createCorrelationData( tImg )

    if False:
        createBinaryCorrelationData( tImg )

    # read all files
    hScores = readScores( 'scoreFile.txt' )
    pScores = readScores( 'pCorrScores.txt' )
    cScores = readScores( 'mCorrScores.txt' )
    bScores = readScores( 'mBinaryCorrScores.txt' )
    dScores = readScores( 'pixelDiffScores.txt' )

    allScores = []
    allScores.append( hScores )
    allScores.append( pScores )
    allScores.append( cScores )
    allScores.append( bScores )
    allScores.append( dScores )

    newScores = filterScores3( allScores, pScores, 0, 0.75 )

    hScores = newScores[0]
    pScores = newScores[1]
    cScores = newScores[2]
    bScores = newScores[3]
    dScores = newScores[4]

    print("Checking shapes")
    for l in newScores:
        print(l.shape)

    pltList = findBestWeights2( hScores, cScores, bScores, dScores )


    corrVal = pltList[:,:,0].flatten()
    w1Val = pltList[:,:,1].flatten()
    w2Val = pltList[:,:,2].flatten()

    maxCorr = np.amax( corrVal )
    iMax = np.argmax( corrVal )
    w1Max = w1Val[iMax]
    w2Max = w2Val[iMax]
    w3Max = 1 - w1Max - w2Max

    createHeatPlot( w1Val, w2Val, corrVal, 'plot_weights.png', 'Max Corr: %f \nw1: %0.2f w2: %0.2f w3 %0.2f'%(maxCorr,w1Max,w2Max, w3Max) )

    eVal, eVectors = getEigenStuff( hScores, cScores, bScores, dScores )

    wScores = w1Max*cScores + w2Max*bScores + w3Max*dScores

    createHeatPlot( hScores, wScores, pScores, 'plot_8.png', 'Weighted Methods' )

# End main

def getEigenStuff( hScores, cScores, bScores, dScores ):

    nData = len(hScores)

    allData = np.zeros((4,nData))

    allData[0,:] = hScores
    allData[1,:] = cScores
    allData[2,:] = bScores
    allData[3,:] = dScores

    C = np.cov(allData)

    w,v = np.linalg.eig( C )

    print(w)
    print(v)

    return w, v


def findBestWeights2( hScores, cScores, bScores, dScores ):

    nw = 100

    w1List = np.linspace( 0.01, 0.99, nw)

    pltList = np.zeros( ( nw, nw, 4 ))

    wImg = np.zeros(( nw, nw ))

    for i,w1 in enumerate( w1List ):
        w2List = np.linspace( 0.01, 1 - w1, nw )
        
        for j,w2 in enumerate( w2List ):
            w3 = 1 - w1 - w2

            #print( w1 + w2 + w3 )
            pltList[i,j,0] = np.corrcoef( hScores, w1*cScores + w2*bScores + w3*dScores )[0,1]
            pltList[i,j,1] = w1
            pltList[i,j,2] = w2
            pltList[i,j,3] = w3

    return pltList
# End find best weights



def findBestWeights( hScores, dScores, cScores ):

    nw = 100

    w1List = np.linspace( 0.1, 1, nw)

    aW = np.zeros( nw )

    for i,w1 in enumerate(w1List):
        w2 = 1 - w1
        aW[i] = np.corrcoef( hScores, w1*cScores + w2*dScores )[0,1]

    plt.plot( w1List, aW )
    plt.title( "CorrCoef = %f with Weight: %f" % ( np.amax( aW ), w1List[np.argmax( aW )] ) )
    plt.savefig('weights.png')

    return w1, w2
# End find best weights


def filterScores3( allScores, fScores, lowFilter, highFilter ):

    newScores = []
    newFScores = []
    nLists = len(allScores)

    print(nLists)

    for i in allScores:
        newScores.append( [] )

    # Filter out lowly perturbed
    for i in range( len( fScores )):

        if fScores[i] > highFilter or fScores[i] < lowFilter:
            continue

        for j in range( nLists ):
            newScores[j].append( allScores[j][i] )

    finalScores = []
    for i in range( nLists ):
        finalScores.append( np.array( newScores[i] ) )

    nBefore = len( allScores[0] )
    nAfter = len( newScores[0] )

    print("Filtering...\n\tBefore: %d   \n\tAfter: %d" % ( nBefore, nAfter )) 

    return finalScores
# End filter scores



def filterScores2( hScores, mScores1, mScores2, pScores, lowFilter, highFilter ):

    nH = []
    nP = []
    nM = []
    nM2 = []

    nBefore = len( hScores )

    # Filter out lowly perturbed
    for i in range( len( hScores )):

        if pScores[i] > highFilter or pScores[i] < lowFilter:
            continue

        nH.append( hScores[i] )
        nP.append( pScores[i] )
        nM.append( mScores1[i] )
        nM2.append( mScores2[i] )

    hScores = np.array( nH )
    mScores1 = np.array( nM )
    mScores2 = np.array( nM2 )
    pScores = np.array( nP )

    nAfter = len( hScores )
    print("Filtering...\n\tBefore: %d   \n\tAfter: %d" % ( nBefore, nAfter )) 

    return hScores, mScores1, mScores2, pScores
# End filter scores



def filterScores( hScores, mScores, pScores, lowFilter, highFilter ):

    nH = []
    nP = []
    nM = []

    nBefore = len( hScores )

    # Filter out lowly perturbed
    for i in range( len( hScores )):

        if pScores[i] > highFilter or pScores[i] < lowFilter:
            continue

        nH.append( hScores[i] )
        nP.append( pScores[i] )
        nM.append( mScores[i] )

    hScores = np.array( nH )
    mScores = np.array( nM )
    pScores = np.array( nP )

    nAfter = len( hScores )
    print("Filtering...\n\tBefore: %d   \n\tAfter: %d" % ( nBefore, nAfter )) 

    return hScores, mScores, pScores
# End filter scores


def binImg( imgIn, threshold ):

    cpImg = np.copy( imgIn )
    cpImg[cpImg > threshold ] = 255
    cpImg[cpImg <= threshold] = 0

    return cpImg
# end binary image creator


def scaleImg( imgIn ):

    imgType = imgIn.dtype
    
    nImg = cv2.resize( imgIn, None, fx = imgScale, fy=imgScale )

    
    nImg = np.uint8( nImg )
    nType = nImg.dtype

    return nImg

# end scaleImg


def createBinaryCorrelationData( tImg ):

    mImgs, iImgs = getImgs()

    pScores = createBinaryCorrPScores( mImgs, iImgs )
    mScores = createBinaryCorrMScores( mImgs, tImg )

    saveScores( pScores, 'pBinaryCorrScores.txt' )
    saveScores( mScores, 'mBinaryCorrScores.txt' )

def createBinaryCorrMScores( mImgs, tImg ):

    mScores = np.zeros( len( mImgs ))

    tImg = scaleImg( tImg )
    tB = np.sum( tImg )
    tAr = tImg.flatten()

    for i,mImg in enumerate(mImgs):

        mImg = binImg( mImg, 45 )

        # Resize image to quicken computation speed
        mImg = scaleImg( mImg )

        # flatten to 1d for correlation
        mAr = mImg.flatten()

        mVal = np.corrcoef( mAr, tAr )[0,1]

        mScores[i] = mVal

    return mScores

# end createMScores()

def createBinaryCorrPScores( mImgs, iImgs ):

    pScores = np.zeros( len( mImgs ))

    for i in range( len( mImgs )):

        mImg = mImgs[i]
        iImg = iImgs[i]

        mImg = binImg( mImg, 45 )
        iImg = binImg( iImg, 45 )

        mAr = mImg.flatten()
        iAr = iImg.flatten()

        pVal = np.corrcoef( mAr, iAr )[0,1]

        pScores[i] = pVal

    return pScores

# end create

def createCorrelationData( tImg ):

    mImgs, iImgs = getImgs()

    pScores = createCorrPScores( mImgs, iImgs )
    mScores = createCorrMScores( mImgs, tImg )

    saveScores( pScores, 'pCorrScores.txt' )
    saveScores( mScores, 'mCorrScores.txt' )


def createCorrMScores( mImgs, tImg ):

    mScores = np.zeros( len( mImgs ))

    tImg = scaleImg( tImg )
    tB = np.sum( tImg )
    tAr = tImg.flatten()

    for i,mImgList in enumerate(mImgs):

        # adjust so mean brightness matches
        mB = np.sum( mImgList )
        mImg = mImgList * ( tB / mB )

        # Resize image to quicken computation speed
        mImg = scaleImg( mImg )

        # flatten to 1d for correlation
        mAr = mImg.flatten()

        mVal = np.corrcoef( mAr, tAr )[0,1]

        mScores[i] = mVal

    return mScores

# end createMScores()





def createCorrPScores( mImgs, iImgs ):

    pScores = np.zeros( len( mImgs ))

    for i in range( len( mImgs )):

        mImg = scaleImg( mImgs[i] )
        iImg = scaleImg( iImgs[i] )

        mB = np.sum( mImg )        
        iB = np.sum( iImg )        

        iImg = iImg * (mB/iB)

        mAr = mImg.flatten()
        iAr = iImg.flatten()

        pVal = np.corrcoef( mAr, iAr )[0,1]

        pScores[i] = pVal

    return pScores

# end createPScores()

def createHeatPlot( hScores, mScores, pScores, saveLoc, titleName ):

    plt.clf()


    pMin = np.amin( pScores )
    pMax = np.amax( pScores )

    cMap = plt.cm.get_cmap('RdYlBu_r')
    #plot = plt.scatter( hScores, mScores, c=pScores, vmin = pMin, vmax = 1, s=10, cmap=cMap)
    plot = plt.scatter( hScores, mScores, c=pScores, vmin = pMin, vmax = pMax, s=10, cmap=cMap)
    cBar = plt.colorbar(plot)

    plt.title(titleName)
    plt.xlabel("Human Scores")
    plt.ylabel("Correlation with Target")

    ax = plt.gca()
    ax.set_facecolor("xkcd:grey")

    cBar.set_label("Correlation with Unperturbed image")

    plt.savefig( saveLoc )

# End plot creation


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

        if mVal < 0.0: mVal = 0.0

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
