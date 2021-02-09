'''
    Author:     Matthew Ogden
    Created:    29 Oct 2019
    Altered:    
Description:    Meant to take the target image, rotate and resize to match an image parameter file
'''

from sys import \
        exit, \
        argv

from os import \
        path,\
        listdir

import numpy as np
import cv2


printAll = True
mainDir = ''

def adjAll( argList ):
    
    endEarly = readArg( argList )

    mainFolders = listdir( mainDir )

    for s in mainFolders:
        sDir = mainDir + s + '/'
        adjSdss( sDir )


def adjSdss(sdssDir):

    print('sdssDir: %s' % sdssDir)

    paramDir = sdssDir + 'sdssParameters/'

    if not path.exists( paramDir ):
        print("sdssDir not found: %s" % paramDir)
        return

    imgParamLoc = paramDir + 'param_v3_default.txt'

    g, imgParam = readFile( imgParamLoc )
    
    fromC = getTargetCenters( paramDir + 'parameters.txt' )

    if len( fromC ) == 0:
        print("Failed: %s" % sdssDir )
        return
    
    tImg = cv2.imread( paramDir + 'target_zoo.png' )

    writeNewParam( imgParamLoc, imgParam, fromC, tImg.shape )


# End main

def writeNewParam( paramLoc, imgParam, fromC, tShape ):

    print(tShape)

    l1, l2, x1, y1, x2, y2 = fromC

    l1Found = False
    l2Found = False

    newParam = []

    for i, l in enumerate(imgParam):

        if 'image_rows' in l:
            nl = 'image_rows %d\n' % tShape[0]
            newParam.append(nl)

        elif 'image_cols' in l:
            newParam.append( 'image_cols %d\n' % tShape[1] )

        elif 'galaxy_1_center' in l:
            newParam.append( 'galaxy_1_center %s %s\n' % ( x1, y1 ) )

        elif 'galaxy_2_center' in l:
            newParam.append( 'galaxy_2_center %s %s\n' % ( x2, y2 ) )

        elif 'galaxy_1_luminosity' in l:
           newParam.append('galaxy_1_luminosity %s\n' % l1 )
           l1Found = True

        elif 'galaxy_2_luminosity' in l:
           newParam.append('galaxy_2_luminosity %s\n' % l2 )
           l2Found = True

        else:
            newParam.append( l )

    if not l1Found:
        newParam.append('galaxy_1_luminosity %s\n' % l1 )

    if not l2Found:
        newParam.append('galaxy_2_luminosity %s\n' % l2 )

    oFile = open( paramLoc, 'w' )

    for l in newParam:
        #print(l)
        oFile.write(l)

    oFile.close()


def getTargetCenters( pLoc ):
    print("Getting target centers")
    g, pFile = readFile( pLoc )

    for l in pFile:

        try:
            if 'primary_luminosity' in l:
                lC = l.strip().split()
                l1 = lC[1]

            elif 'secondary_luminosity' in l:
                lC = l.strip().split()
                l2 = lC[1]

            elif 'target_zoo.png' in l:
                
                lC = l.strip().split()

                x1 = lC[1]
                y1 = lC[2]
                x2 = lC[3]
                y2 = lC[4]
        except:
            return []

    return [ l1, l2, x1, y1, x2, y2 ]



# end get t centers

def getParamInfo( pLoc ):
    print("Getting img param centers")
    good, sFile = readFile( pLoc )

    pCenters = np.zeros((2,2))
    pSize = np.zeros(2)

    for l in sFile:
        l = l.strip()

        if 'galaxy_1_center' in l:
            lC = l.split()
            pCenters[0,0] = int(lC[1])
            pCenters[0,1] = int(lC[2])

        elif 'galaxy_2_center' in l:
            lC = l.split()
            pCenters[1,0] = int(lC[1])
            pCenters[1,1] = int(lC[2])

        elif 'image_rows' in l:
            pSize[0] = l.split()[1]

        elif 'image_cols' in l:
            pSize[1] = l.split()[1]

    print(pCenters)
    print(pSize)

    return pCenters, pSize

# end get t centers

def readArg(argList):

    global printAll, mainDir, sdssDir

    endEarly = False

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

    # Check if input arguments were valid

    return endEarly

# End reading command line arguments

def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return False, []
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return False, []

    else:
        inList = list(inFile)
        inFile.close()
        return True, inList

# End simple read file


def overLapTarget( image, iC, target, tC):

    if printAll:
        print('About to overlap images')

    #Note: Warping images via opencv requires third point

    # Calculate points to move galaxies
    toPoint = np.zeros((3,2))
    toPoint[0,:] = [ toShape[0]/3 , toShape[1]/2 ]
    toPoint[1,:] = [ toShape[0]*2/3 , toShape[1]/2 ]
    toPoint[2,0] = int( toPoint[0,0] + ( toPoint[0,1] - toPoint[1,1] ) )
    toPoint[2,1] = int( toPoint[0,1] + ( toPoint[1,0] - toPoint[0,0] ) )


    # Create third point for image
    #print(iC)
    nIC = np.zeros((3,2))
    ix = int( iC[0,0] + ( iC[0,1] - iC[1,1] ) )
    iy = int( iC[0,1] + ( iC[1,0] - iC[0,0] ) )

    nIC[0:2,:] = iC
    nIC[2,:] = [ ix, iy ]

    # Warp image
    warpMat = cv2.getAffineTransform( np.float32(nIC), np.float32(toPoint) )
    newImg = cv2.warpAffine(image, warpMat, toShape)


    # Create third point for target 
    nTC = np.zeros((3,2))
    tx = int( tC[0,0] + ( tC[0,1] - tC[1,1] ) )
    ty = int( tC[0,1] + ( tC[1,0] - tC[0,0] ) )

    nTC[0:2,:] = tC
    nTC[2,:] = [ tx, ty ]

    # Calcuate warp matrix
    warpMat2 = cv2.getAffineTransform( np.float32(nTC), np.float32(toPoint) )
    newTarget = cv2.warpAffine(target, warpMat2, toShape)

    if printAll:
        print('Overlapped images')

    return newImg, newTarget
# end overLap

# Run main after declaring functions
if __name__=='__main__':
    argList = argv
    #main(argList)
    adjAll(argList)
