'''
    Author:     Matthew Ogden
    Created:    12 July 2019
    Altered:    12 July 2019
Description:    This python code is intended to directly compare the pixel by pixel 
                values between a model image and a target image using OpenCV.

'''


import os
import sys
import re
import cv2
import numpy as np



# Global input arguments
imageLoc = ''
imgInfoLoc = ''
imgParam = ''
runDir = ''

targetLoc = ''
targetInfoLoc = ''
targetCenterString = ''

scoreLoc = ''
humanScore = ''

checkGalCenters = False



returnVal = False
printAll = True


def main():

    endEarly = readArg()

    if endEarly:
        print('\nExiting pixel_comparison...\n')
        sys.exit(-1)

   # Open and read image Info file
    try:
        imgInfoFile = open(imgInfoLoc, 'r')
        sdssName, genName, runNum, imgCenters = readImgInfoFile( imgInfoFile )
        imgInfoFile.close()
    except:
        print('Failed to open and read image info file at \'%s\'' % imgInfoLoc)
        sys.exit(-1)

    # get target info
    try:
        targetInfoFile = open( targetInfoLoc, 'r' )
        targetCenters = readTargetInfoFile( targetInfoFile )
        targetInfoFile.close()
    except:
        print('Failed to open and read target info file at \'%s\'' % targetInfoLoc)
        sys.exit(-1)
    

    # Check if images exist
    try:
        image = cv2.imread(imageLoc)
        target = cv2.imread(targetLoc)

        # Quick conversion needed when reading from info file
        targetCenters[1,1] = target.shape[1] - targetCenters[1,1]
    except:
        print('Failed to open an image at \'%s\' or \'%s\''% (targetLoc,imageLoc))
        sys.exit(-1)

    # draw circles on images to check if centers are correct if needed
    if checkGalCenters:
        drawGalCenters( image, target, imgCenters, targetCenters )

    # Create new image that overlaps target image
    newImg = overLapTarget( image, imgCenters, target, targetCenters) 

    # Do comparison
    # Working Here...    Will need to load calibrated image in greyscale
    
    if newImg.shape != target.shape:
        print('image and target have different shapes')
        print('Exiting...\n')
        #sys.exit(-1)

    score = direct_subtract(newImg,target)

    # creat score file if not present
    if not os.path.isfile( scoreLoc ):
        scoreFile = open( scoreLoc, 'w')
        scoreFile.write('sdss,generation,run,target_image,model_image,image_parameter,human_score,comparison_method,machine_score\n')
    else:
        scoreFile = open( scoreLoc, 'a')

    sLine = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ( sdssName, genName, runNum, targetLoc, imageLoc, imgParam, humanScore, 'pixel_subtraction_v0', score)

    scoreFile.write(sLine)

    scoreFile.close()

    #sdssName, genName, runNum, imgCenters = readImgInfoFile( imgInfoFile )
 
# End main

def direct_subtract( img1, img2 ):

    return 0
# end direct_subtract
    
def overLapTarget( image, iC, target, tC):


    nIC = np.zeros((3,2))

    # Create third point for 
    tx = int( iC[0,0] + ( iC[0,1] - iC[1,1] ) )
    ty = int( iC[0,1] + ( iC[1,0] - iC[0,0] ) )

    nIC[0:2,:] = iC
    nIC[2,0] = tx
    nIC[2,1] = ty

    nTC = np.zeros((3,2))

    # Create third point for target 
    tx = int( tC[0,0] + ( tC[0,1] - tC[1,1] ) )
    ty = int( tC[0,1] + ( tC[1,0] - tC[0,0] ) )

    nTC[0:2,:] = tC
    nTC[2,0] = tx
    nTC[2,1] = ty

    # Calcuate warp matrix
    warpMat = cv2.getAffineTransform( np.float32(nIC), np.float32(nTC) )

    # Create new model image that would overlap target
    #print(target.shape[0:2])
    newImg = cv2.warpAffine(image, warpMat, target.shape[0:2])

    return newImg
# end overLap


def drawGalCenters( image, target, iCenters, tCenters):
    print('drawing gal centers')
    
    cv2.circle( image, ( int(iCenters[0,0]), int(iCenters[0,1]) ), 15, (255,0,0), 3)
    cv2.circle( image, ( int(iCenters[1,0]), int(iCenters[1,1]) ), 15, (0,255,0), 3)

    cv2.imwrite( runDir + 'img_circle.png', image)

    cv2.circle( target, ( int(tCenters[0,0]), int(tCenters[0,1]) ), 15, (255,0,0), 3)
    cv2.circle( target, ( int(tCenters[1,0]), int(tCenters[1,1]) ), 15, (0,255,0), 3)

    cv2.imwrite( runDir + 'target_circle.png', target)

    print('Created iamges with circle at %s' % runDir)
    print(iCenters)
    sys.exit(1)

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
        sys.exit(-1)
    '''

    return tCenters

#end read target info



def readImgInfoFile( imgInfoFile ):

    sdssName = genName = runNum = ''
    imgCenters = np.zeros(4)

    for l in imgInfoFile:
        l = l.strip()
        #print(l)


        if 'sdss_name' in l:
            sdssName = l.split()[1]

        elif 'generation' in l:
            genName = l.split()[1]

        elif 'run_number' in l:
            runNum = l.split()[1]

        elif imgParam in l:
            pLine = l.split()

            try:
                for i in range(4):
                    imgCenters[i] = int( pLine[i+1] ) 

                imgCenters = imgCenters.reshape((2,2))
                #print(imgCenters)

            except:

                print('Parameter line \'%s\' not correct format' % l )
                print('Expected format like \'param0000 100 100 400 400\'')
                print('Exiting...\n')
                sys.exit(-1)


    # Check if all values were found
    if sdssName == '' or runNum == '' or genName == '' or imgCenters == np.zeros(4):
        print('Failed to get all information from info file')
        print('Exiting...\n')
        sys.exit(-1)

    else:
        return sdssName, genName, runNum, imgCenters
# end read img info file



def readArg():

    global imageLoc, imgInfoLoc, imgParam, runDir
    global targetLoc, targetInfoLoc, targetCenterString
    global scoreLoc, returnVal, allMethods, checkGalCenter, humanScore

    for i,arg in enumerate(sys.argv):
    
        endEarly = False

        # move to next argument if not identified with argument specifier
        if arg[0] != '-':
            continue

        elif arg == '-image':
            imageLoc = sys.argv[i+1]
            # There's a better way to do this, But i don't have internet
            dirParse = imageLoc.split('/')
            runDir = ''
            for i in range( len( dirParse ) - 1 ):
                runDir = runDir + dirParse[i] + '/'

        elif arg == '-imgInfo':
            imgInfoLoc = sys.argv[i+1]

        elif arg == '-imgParam':
            imgParam = sys.argv[i+1]

        elif arg == '-target':
            targetLoc = sys.argv[i+1]

        elif arg == '-targetInfo':
            targetInfoLoc = sys.argv[i+1]

        elif arg == '-targetCenter':
            targetCenters = sys.argv[i+1]

        elif arg == '-score':
            scoreLoc = sys.argv[i+1]

        elif arg == '-all':
            allMethods = True

        elif arg == '-center':
            checkGalCenter = True

        elif arg == 'humanScore':
            humanScore = sys.argv[i+1]

        # Future implemenation.  Not yet working
        # Meant for returning score values outside program
        elif arg == '-return':
            returnVal = True
            print('Returning value not yet working.  Exiting...')
            sys.exit(-1)


    # Check if arguments are valid

    # Check if image exists
    if not os.path.isfile(imageLoc):
        print( 'image at \'%s\' not found' % imageLoc )
        endEarly = True

    # Chech if information passed for image exists
    if not os.path.isfile( imgInfoLoc ):
        print( 'Image info file at \'%s\' not found' % imgInfoLoc )
        endEarly = True

    if imgParam == '':
        # Check if image file has param in name
        imgParam = re.findall( r'param\d+', imageLoc)

        if len(imgParam) != 1:
            print('Please specify image parameter')
            endEarly = True
        else:
            imgParam = imgParam[0]


    # Check if target image exists
    if not os.path.isfile(targetLoc):
        print( 'Target image at \'%s\' not found' % targetLoc )
        endEarly = True

    # Check if imformation passed for target
    if not os.path.isfile( targetInfoLoc ) and targetCenters == '':
        print( 'Target info file at \'%s\' not found' % targetInfoLoc )
        print( 'No galaxy centers given for target image')

        endEarly = True

    return endEarly

# end reading arguments

main()
