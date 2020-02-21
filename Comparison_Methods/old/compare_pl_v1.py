'''
    Author:     Matthew Ogden
    Created:    12 July 2019
    Altered:    13 Oct 2019
Description:    Primary code for preparing a target and model image,
                 sending them to comparison methods for scoring,
                 and recording their scores.  Designed to be integrated
                 with a larger pipeline from model to score.

'''

from os import \
        path,\
        listdir, \
        system

from sys import \
        exit, \
        argv
        
from re import findall
from datetime import datetime
import cv2
import numpy as np

import methods.pixel_difference as pixMod
import image_prep.image_prep_v1 as imgPrepMod

# Global input arguments
printAll = True
checkGalCenter = False

runDir = ''
paramLoc = ''

targetLoc = ''
targetInfoLoc = ''

overWriteScore = False
printScore = False
writeScore = True

userComment = 'Testing pipeline v1. Oct 13, 2019'

writeDiffImage = False

def compare_pl_v1( argList ):

    print("In compare!")
    imgPrepMod.check()

    endEarly = nReadArg( argList )

# new read Artuments
def nReadArg( argList ):

    global printAll, checkGalCenter 
    global overWriteScoreFile, printScore, appendScore
    global runDir, paramLoc

    # TODO!   Later....  whenever needed
    global targetLoc, targetInfoLoc

    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir += '/'

        elif arg == '-paramLoc':
            paramLoc = argList[i+1]

        elif arg == '-overWriteScore':
            overWriteScore = True

        elif arg == '-center':
            checkGalCenter = True

        # Not used for now
        elif arg == '-target':
            targetLoc = argList[i+1]

        elif arg == '-targetInfo':
            targetInfoLoc = argList[i+1]

        elif arg == '-all':
            comList = []

        elif arg == '-toShape':
            endEarly = readShape(argList[i+1])

        elif arg == '-writeDiffImage':
            writeDiffImage = True

        elif arg == '-methodName':
            methodName = argList[i+1]

    # Check if arguments are valid

    if printAll:
        print('\nChecking validity of arguments')

    # Check if information passed for target
    if not path.isfile( targetInfoLoc ):
        print( 'Target info file at \'%s\' not found' % targetInfoLoc )
        endEarly = True

    # Check if any comparison method was chosen
    if methodName == '' and not diffMethod and not diffSqMethod and not diffSqNonZeroMethod and not diffNonZeroMethod:
        print('No comparison methods selected')
        endEarly = True


    return endEarly

# end reading arguments



def main():

    endEarly = readArg()

    if printAll:
        print('\nRun directory: %s' % runDir)
        print('Image Loc: %s' % imageLoc)
        print('Info Loc: %s' % imgInfoLoc)
        print('Target Loc: %s' % targetLoc)
        print('Target info Loc: %s' % targetInfoLoc)
        if diffMethod:
            print('Using method: pixel_difference')
        if diffSqMethod:
            print('Using method: pixel_difference_squared')
        print('')


    if endEarly:
        print('\nExiting pixel_comparison...\n')
        exit(1)

   # Open and read image Info file
    try:
        imgInfoFile = open(imgInfoLoc, 'r')
        readImgInfoFile( imgInfoFile )
        imgInfoFile.close()
    except:
        print('Failed to open and read image info file at \'%s\'' % imgInfoLoc)
        exit(-1)

    # get target info
    try:
        targetInfoFile = open( targetInfoLoc, 'r' )
        targetCenters = readTargetInfoFile( targetInfoFile )
        targetInfoFile.close()
    except:
        print('Failed to open and read target info file at \'%s\'' % targetInfoLoc)
        exit(-1)


    # Check if images exist
    try:
        image = cv2.imread(imageLoc,0)
        target = cv2.imread(targetLoc,0)

        # Quick conversion needed when reading from info file
        targetCenters[1,1] = target.shape[1] - targetCenters[1,1]
    except:
        print('Failed to open an image at \'%s\' or \'%s\''% (targetLoc,imageLoc))
        exit(-1)

    # draw circles on images to check if centers are correct if needed
    if checkGalCenter:
        drawGalCenters( image, target, imgCenters, targetCenters )
        exit(-1)

    # Rotate and translate images to overlap
    image, target = overLapTarget( image, imgCenters, target, targetCenters) 
    cv2.imwrite( 'target.png' , target)


    # Final check to see if images are the same size
    if image.size != target.size:
        print('image and target have different shapes')
        print('Exiting...\n')
        exit(-1)


    if methodName == 'diffSqNonZero':
        methodFunc = pixMod.diffSquaredNonZero
    elif methodName == 'diff':
        methodFunc = pixMod.pixel_difference

    '''
    if diffMethod:
        methodFunc = pixMod.pixel_difference
    elif diffSqMethod:
        methodFunc = pixMod.pixel_difference_squared
    elif diffSqNonZeroMethod:
        methodFunc = pixMod.diffSquaredNonZero
    elif diffNonZeroMethod:
        methodFunc = pixMod.diffNonZero
    '''

    score, methodFullName, diffImg = methodFunc( image, target )

    if printAll:
        print('Score: %f  Method: %s' % ( score, methodFullName ))
        print('Max: %f Min: %f' % ( np.amax( diffImg ), np.amin( diffImg) ) )

    if True or writeDiffImage:
        cv2.imwrite( runDir + '%s.png' % methodFullName, diffImg)

    if printScore:
        print(score)

    # prepare score file
    if writeScore:
        writeScoreFile(scoreLoc, score, methodFullName)



# End main

def writeScoreFile(scoreLoc, score, methodFullName ):

    #scoreFile.write('sdss,generation,run,zoo_model_data,human_score,target_image,model_image,image_parameter,comparison_method,machine_score\n')

    # Delete score file if overwriting
    if overWriteScore:
        if printAll:
            print('Overwriting score file %s' % scoreLoc)

        if path.isfile( scoreLoc):
            system('rm %s' % scoreLoc)

    # create score file if not present
    if not path.isfile( scoreLoc ):
        scoreFile = open( scoreLoc, 'w')

        # Write Header
        scoreFile.write('sdss,generation,run,zoo_model_name,zoo_model_data,human_score,target_image,model_image,image_parameter,comparison_method,machine_score,user_comment,date_time\n')

    else:
        # Else append
        scoreFile = open( scoreLoc, 'a')

    sLine  =  '%s' % sdssName
    sLine += ',%s' % genName
    sLine += ',%s' % runName
    sLine += ',%s' % modelName
    sLine += ',"%s"' % modelData
    sLine += ',%s' % humanScore
    sLine += ',%s' % targetLoc
    sLine += ',%s' % imageLoc
    sLine += ',%s' % imgParam
    sLine += ',%s' % methodFullName
    sLine += ',%s' % score
    sLine += ',"%s"' % userComment
    sLine += ',%s' % datetime.now()
    
    sLine += '\n' 

    scoreFile.write(sLine)
    scoreFile.close()

# End write score to scoreFile


def readRunDir( runDir ):

    global imageLoc, imgInfoLoc, scoreLoc

    try:
        dirItems = listdir( runDir )
    except:
        print('Failed to read run directory \'%s\'' % runDir)
    else:
        
        for f in dirItems:

            fullPath = runDir + f

            if 'model.png' in f:
                imageLoc = fullPath

            if 'info.txt' in f:
                imgInfoLoc = fullPath

            if 'scores.csv' in f:
                scoreLoc = fullPath

        if scoreLoc == '':
            scoreLoc = runDir + 'scores.csv'


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


def drawGalCenters( image, target, iCenters, tCenters):
    if printAll:
        print('Drawing gal centers')
        print(iCenters)
        print(tCenters)
    
    cv2.circle( image, ( int(iCenters[0,0]), int(iCenters[0,1]) ), 15, (255,0,0), 3)
    cv2.circle( image, ( int(iCenters[1,0]), int(iCenters[1,1]) ), 15, (0,255,0), 3)

    cImgLoc = runDir + 'img_center.png'
    cv2.imwrite( cImgLoc, image)
    print('Wrote image centers at %s' % cImgLoc)

    cv2.circle( target, ( int(tCenters[0,0]), int(tCenters[0,1]) ), 15, (255,0,0), 3)
    cv2.circle( target, ( int(tCenters[1,0]), int(tCenters[1,1]) ), 15, (0,255,0), 3)

    cTargetLoc = runDir + 'target_center.png'
    cv2.imwrite( cTargetLoc, target)
    print('Wrote target centerst at %s' % cTargetLoc)

#end draw gal centers


def readTargetInfoFile(targetInfoFile):

    tCenters = np.zeros((2,2))

    for l in targetInfoFile:
        l = l.strip()

        if 'pxc=' in l:
            tCenters[0,0] = int( float( l.split('=')[1] ) ) 

        if 'pyc=' in l:
            tCenters[0,1] = int( float( l.split('=')[1] ) )

        if 'sxc=' in l:
            tCenters[1,0] = int( float( l.split('=')[1] ) )

        if 'syc=' in l:
            tCenters[1,1] = int( float( l.split('=')[1] ) )

    # End looping through target file


    # this trouble shooting method doesn't work.  May work on later
    '''
    if np.compare( np.zeros((2,2)), tCenters):
        print('Failed to find center point in target info file')
        print('Exiting...\n')
        exit(-1)
    '''

    return tCenters

#end read target info



def readImgInfoFile( imgInfoFile ):

    global sdssName, genName, runName, modelName, modelData, humanScore, imgCenters

    if printAll:
        print('Reading info file %s' % imgInfoLoc)

    foundCenters = False
    imgCenters = np.zeros(4)

    for l in imgInfoFile:
        l = l.strip()

        if 'sdss_name' in l:
            sdssName = l.split()[1]

        elif 'generation' in l:
            genName = l.split()[1]

        elif 'run_number' in l:
            runName = l.split()[1]

        elif 'model_name' in l:
            modelName = l.split()[1]

        elif 'model_data' in l:
            modelData = l.split()[1]

        elif 'human_score' in l:
            humanScore = l.split()[1]

        elif imgParam in l:
            pLine = l.split()

            try:
                for i in range(4):
                    imgCenters[i] = int( pLine[i+1] ) 

                foundCenters = True
            except:

                print('Parameter line \'%s\' not correct format' % l )
                print('Expected format like \'param0000 100 100 400 400\'')
                print('Exiting...\n')
                exit(-1)

    imgCenters = imgCenters.reshape((2,2))
    if printAll:
        print('Found: %s %s %s'% ( sdssName, runName, genName) )
        print(imgCenters)


    # Check if all values were found
    if sdssName == '' or runName == '' or genName == '' or not foundCenters:
        print('Failed to get all information from info file')
        print('Exiting...\n')
        exit(-1)

# end read img info file

def readShape( shapeIn ):

    global toShape

    try:
        shapeVals = shapeIn.split(',')
        toShape[0] = int(shapeVal[0])
        toShape[1] = int(shapeVal[1])
        return False
    except:
        print('Failed to read in image shape %s' % shapeIn)
        print("Expecting '1024,1024' format")

        return True


def readArg():

    global printAll
    global runDir, imgParam, imageLoc, imgInfoLoc, scoreLoc, overWriteScore
    global checkGalCenter, writeDiffImages
    global printScore, writeScore, methodName

    # TODO when needed
    global targetLoc, targetInfoLoc
    global toPoint
    global diffMethod, diffSqMethod, featMethods, diffSqNonZeroMethod, diffNonZeroMethod

    endEarly = False

    for i,arg in enumerate(argList):
    

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFile = argList[i+1]
            argList = readArgFile( argList, argFile )

        elif arg == '-noprint':
            printAll = False

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir += '/'
            readRunDir(runDir)

        elif arg == '-image':
            imageLoc = argList[i+1]

        elif arg == '-imgInfo':
            imgInfoLoc = argList[i+1]

        elif arg == '-imgParam':
            imgParam = argList[i+1]
        
        elif arg == '-score':
            scoreLoc = argList[i+1]

        elif arg == '-target':
            targetLoc = argList[i+1]

        elif arg == '-targetInfo':
            targetInfoLoc = argList[i+1]

        elif arg == '-center':
            checkGalCenter = True

        elif arg == '-humanScore':
            humanScore = argList[i+1]

        elif arg == '-zooModel':
            zooModel = argList[i+1]

        elif arg == '-all':
            diffMethod = True
            diffSqMethod = True
            featMethods = True

        elif arg == '-diff':
            diffMethod = True

        elif arg == '-diffSq':
            diffSqMethod = True

        elif arg == '-diffSqNonZero':
            diffSqNonZeroMethod = True

        elif arg == '-diffNonZero':
            diffNonZeroMethod = True

        elif arg == '-toShape':
            endEarly = readShape(argList[i+1])

        elif arg == '-overWriteScore':
            overWriteScore = True

        elif arg == '-writeDiffImage':
            writeDiffImage = True

        elif arg == '-printScore':
            printScore = True

        elif arg == '-noScore':
            writeScore = False

        elif arg == '-methodName':
            methodName = argList[i+1]

    # Check if arguments are valid

    if printAll:
        print('\nChecking validity of arguments')

    # Check if image exists
    if not path.isfile(imageLoc):
        print( 'Model image at \'%s\' not found' % imageLoc )
        endEarly = True

    # Chech if information passed for image exists
    if not path.isfile( imgInfoLoc ):
        print( 'Image info file at \'%s\' not found' % imgInfoLoc )
        endEarly = True

    if imgParam == '':
        # Check if image file has param in name
        imgParam = findall( r'param\d+', imageLoc)

        if len(imgParam) != 1:
            print('Please specify image parameter')
            endEarly = True
        else:
            imgParam = imgParam[0]


    # Check if target image exists
    if not path.isfile(targetLoc):
        print( 'Target image at \'%s\' not found' % targetLoc )
        endEarly = True

    # Check if imformation passed for target
    if not path.isfile( targetInfoLoc ):
        print( 'Target info file at \'%s\' not found' % targetInfoLoc )
        endEarly = True

    # Check if any comparison method was chosen
    if methodName == '' and not diffMethod and not diffSqMethod and not diffSqNonZeroMethod and not diffNonZeroMethod:
        print('No comparison methods selected')
        endEarly = True


    return endEarly

# end reading arguments

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

if __name__ == '__main__':

    argList = argv
    compare_pl_v1( argList )

