'''
     Author:    Matthew Ogden
    Created:    31 Oct 2019
    Altered:    
Description:    Created for simple comparison of sdss images
'''

import numpy as np
import cv2

def test():
    print("Inside machineMethods.py")

def scoreOverLap( img1, img2, h ):

    score = -1

    i1 = np.copy( img1 )
    i2 = np.copy( img2 )

    i1[ i1 <  h ] = 0
    i1[ i1 >= h ] = 1

    i2[ i2 <  h ] = 0
    i2[ i2 >= h ] = 1

    bImg = i1 + i2
    bImg[ bImg <  2 ] = 0
    bImg[ bImg >= 2 ] = 1

    x = np.sum( i1 )
    y = np.sum( i2 )
    z = np.sum( bImg )

    score = ( z / ( x + y - 1.0*z ) )

    return score


def scoreCorrelation( img1, img2 ):

    score = -1

    if len( img1.shape ) > 1:
        ar1 = img1.flatten()
    else:
        ar1 = img1

    if len( img2.shape ) > 1:
        ar2 = img2.flatten()
    else:
        ar2 = img2

    score = np.corrcoef( ar1, ar2 )[0,1]

    return score


def binImg( imgIn, threshold ):

    cpImg = np.copy( imgIn )
    cpImg[ cpImg >= threshold] = 255
    cpImg[ cpImg < threshold] = 0

    return cpImg



def scoreBinaryCorrelation( img1, img2, threshold ):

    score = -1
    
    img1 = binImg( img1, threshold )

    score = scoreCorrelation( img1, img2 )

    return score



# end createBinaryCorrelation()


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


def scoreAbsDiff( img1, img2 ):

    score = -1

    # adjust so mean brightness matches

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score
    return score
# End absDiff




def scoreAbsDiff2( img1, img2 ):

    score = -1

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score

    # arbitrary adjustment to make score more human readable 
    score = 10 * (score - 0.9)
    if score < 0.0: score = 0.0

    return score

# end createMScores()


def scoreAbsDiff3( img1, img2 ):

    score = -1

    dImg = np.abs( img1 - img2 )
    score = np.sum( dImg ) / dImg.size / 255
    score = 1 - score

    # arbitrary adjustment to make score more human readable 
    score = 5 * (score - 0.8)
    if score < 0.0: score = 0.0

    return score

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

    global printAll, imgDir, everyN, plotDir, scoreDir

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

        elif arg == '-plotDir':
            plotDir = argList[i+1]

        elif arg == '-scoreDir':
            scoreDir = argList[i+1]

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
if __name__=='__main__':

    argList = argv
    endEarly = readArg()

    if printAll: 
        print("ImgDir  : %s" % imgDir )
        print("plotDir : %s" % plotDir )

    if endEarly: exit(-1)

    new_main()
    #runSpread()
    #make_plots()
    #make_plots2()
    #main_old()

