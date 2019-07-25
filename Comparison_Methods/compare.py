'''
    Author:     Matthew Ogden
    Created:    12 July 2019
    Altered:    22 July 2019
Description:    This python code is intended to directly compare the pixel by pixel 
                values between a model image and a target image using OpenCV.

'''


from os import \
        path,\
        listdir, \
        system

from sys import \
        exit, \
        argv
        
from re import findall

import cv2
import numpy as np


import methods.pixel_difference as pixMod
import methods.cv_features as featMod

# Global input arguments
printAll = True

runDir = ''
sdssName = ''
genName = ''
runName = ''
zooModel = ''
humanScore = ''
userComment = 'Inital creation and testing of comparison code'

imageLoc = ''
imgInfoLoc = ''
scoreLoc = ''
imgParam = ''

overWriteScore = False

targetLoc = ''
targetInfoLoc = ''

checkGalCenter = False

writeDiffImage = False

diffMethod = False
diffSqMethod = False
featMethods = False

toShape = ( 1024, 1024 )


def main():

    endEarly = readArg()

    if printAll:
        print('Run directory: %s' % runDir)
        print('Image Loc: %s' % imageLoc)
        print('Info Loc: %s' % imgInfoLoc)
        print('Target Loc: %s' % targetLoc)
        print('Target info Loc: %s' % targetInfoLoc)
        if diffMethod:
            print('Using method: pixel_difference')
        if diffSqMethod:
            print('Using method: pixel_difference_squared')


    if endEarly:
        print('\nExiting pixel_comparison...\n')
        exit(-1)

   # Open and read image Info file
    try:
        imgInfoFile = open(imgInfoLoc, 'r')
        global sdssName, genName, runName
        sdssName, genName, runName, imgCenters = readImgInfoFile( imgInfoFile )
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


    # prepare score file
    # Delete score file if overwriting
    if overWriteScore:
        if printAll:
            print('Overwriting score file %s' % scoreLoc)
        system('rm %s' % scoreLoc)

    # creat score file if not present
    if not path.isfile( scoreLoc ):
        scoreFile = open( scoreLoc, 'w')
        scoreFile.write('sdss,generation,run,zoo_model_data,human_score,target_image,model_image,image_parameter,comparison_method,machine_score,user_comment\n')
    else:
        scoreFile = open( scoreLoc, 'a')


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


    # Final check to see if images are the same size
    if image.size != target.size:
        print('image and target have different shapes')
        print('Exiting...\n')
        exit(-1)

    if diffMethod:
        score, methodName, diffImg = pixMod.pixel_difference( image, target )
        writeScore(scoreFile, score, methodName)
        if printAll:
            print('Score: %f  Method: %s' % ( score, methodName ))
            print('Max: %f Min: %f' % ( np.amax( diffImg ), np.amin( diffImg) ) )
        if writeDiffImage:
            cv2.imwrite( runDir + '%s.png' % methodName, diffImg)
    
    if diffSqMethod:
        score, methodName, diffImg = pixMod.pixel_difference_squared( image, target )
        writeScore(scoreFile, score, methodName)
        if printAll:
            print('Score: %f  Method: %s' % ( score, methodName ))
            print('Max: %f Min: %f' % ( np.amax( diffImg ), np.amin( diffImg) ) )
        if writeDiffImage:
            cv2.imwrite( runDir + '%s.png' % methodName, diffImg)

    if featMethods:
        score, methodName, diffImg = featMod.harris_corner_compare( image, target )
        writeScore(scoreFile, score, methodName)
        if printAll:
            print('Score: %f  Method: %s' % ( score, methodName ))
            print('Max: %f Min: %f' % ( np.amax( diffImg ), np.amin( diffImg) ) )
        if writeDiffImage:
            cv2.imwrite( runDir + '%s.png' % methodName, diffImg)
    
    scoreFile.close()

    #sdssName, genName, runNum, imgCenters = readImgInfoFile( imgInfoFile )
 
# End main

def writeScore(scoreFile, score, methodName ):

    #scoreFile.write('sdss,generation,run,zoo_model_data,human_score,target_image,model_image,image_parameter,comparison_method,machine_score\n')

    sLine = '%s' % sdssName
    sLine += ',%s' % genName
    sLine += ',%s' % runName
    sLine += ',%s' % '' #'\0' + zooModel + '\0'  I can't remeber the character 
    sLine += ',%s' % humanScore
    sLine += ',%s' % targetLoc
    sLine += ',%s' % imageLoc
    sLine += ',%s' % imgParam
    sLine += ',%s' % methodName
    sLine += ',%s' % score
    sLine += ',%s\n' % userComment

    scoreFile.write(sLine)


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

def direct_subtract( img1, img2 ):

    return 0
# end direct_subtract
    
def overLapTarget( image, iC, target, tC):

    if printAll:
        print('About to overlap images')

    #Note: Warping images via opencv requires third point

    # Calculate points to move galaxies
    toPoint = np.zeros((3,2))
    toPoint[0,:] = [ toShape[0]/2 , toShape[1]/3 ]
    toPoint[1,:] = [ toShape[0]/2 , toShape[1]*2/3 ]
    toPoint[2,0] = int( toPoint[0,0] + ( toPoint[0,1] - toPoint[1,1] ) )
    toPoint[2,1] = int( toPoint[0,1] + ( toPoint[1,0] - toPoint[0,0] ) )


    # Create third point for image
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

    if printAll:
        print('Reading info file %s' % imgInfoLoc)

    sdssName = ''
    genName = ''
    runName = ''
    
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

        elif imgParam in l:
            pLine = l.split()

            try:
                for i in range(4):
                    imgCenters[i] = int( pLine[i+1] ) 

                imgCenters = imgCenters.reshape((2,2))
                foundCenters = True
                #print(imgCenters)

            except:

                print('Parameter line \'%s\' not correct format' % l )
                print('Expected format like \'param0000 100 100 400 400\'')
                print('Exiting...\n')
                exit(-1)


    # Check if all values were found
    if sdssName == '' or runName == '' or genName == '' or not foundCenters:
        print('Failed to get all information from info file')
        print('Exiting...\n')
        exit(-1)

    else:
        return sdssName, genName, runName, imgCenters
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
    global targetLoc, targetInfoLoc
    global humanScore, zooModel
    global checkGalCenter, writeDiffImage
    global diffMethod, diffSqMethod, featMethods
    global toPoint

    argList = argv

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

        elif arg == '-toShape':
            endEarly = readShape(argList[i+1])

        elif arg == '-overWriteScore':
            overWriteScore = True

        elif arg == '-writeDiffImage':
            writeDiffImage = True

    # Check if arguments are valid

    if printAll:
        print('Check validity of arguments')
    # Check if image exists
    if not path.isfile(imageLoc):
        print( 'image at \'%s\' not found' % imageLoc )
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
    if not diffMethod and not diffSqMethod:
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

main()