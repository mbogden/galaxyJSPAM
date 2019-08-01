'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    
Description:    This is my python version 2 for creating images from spam particle files.
'''

from sys import \
        exit, \
        argv

from os import \
        path,\
        listdir, \
        system

import numpy as np
import cv2

# Global input variables
printAll = True
makeMask = False
overwriteImage = True

runDir = ''
imageLoc = ''

partLoc1 = ''
partLoc2 = ''

infoFile = ''
imgHeaderFound = ''

paramLoc = ''
paramInfo = ''
paramName = ''




def main():

    print('In image_creator_v2')
    global imageLoc

    endEarly = readArg()

    if printAll:
        print('runDir: %s' % runDir)
        print('partLoc1: %s' % partLoc1)
        print('partLoc2: %s' % partLoc2)
        print('paramLoc: %s' % paramLoc)
        print('paramInfo: %s' % paramInfo)

    if endEarly:
        exit(-1)

    if paramLoc != '':
        pName, gSize, gWeight, rConst, normVal, nRows, nCols = readParamFile( paramLoc ) 
    elif paramInfo != '':
        pName, gSize, gWeight, rConst, normVal,nRows, nCols = paramInfo.split() 
    
    else:
        print('You shouldn\'t be seeing this')

    g1iPart, g2iPart, iCenters = readPartFile( partLoc1 )
    g1fPart, g2fPart, fCenters = readPartFile( partLoc2 )


    if imageLoc == '':
        imageLoc = runDir + '%s_model.png' % pName

    if printAll:
        print("Saving image at %s" % imageLoc)

    g1fPart, g2fPart, fCenters = shiftPoints( g1fPart, g2fPart, fCenters, nRows, nCols )

    #simpleWriteImg( runDir + 'simple.png', g1fPart, g2fPart, nRows, nCols )

    dotImg = createDotImg( g1fPart, g2fPart, nRows, nCols, g1iPart[:,3], g2iPart[:,3], rConst )
    #cv2.imwrite( runDir + 'dot.png', dotImg )

    blurImg = cv2.GaussianBlur( dotImg, (gSize, gSize), gWeight )
    cv2.imwrite( imageLoc, blurImg )

    writeParamInfo( infoLoc, pName, fCenters )


# End main

def writeParamInfo( infoLoc, pName, centers ):

    # Check if info
    if not path.isfile( infoLoc ):
        return
    
    else:
        infoFile = open( infoLoc, 'r' )
        infoList = list( infoFile )
        infoFile.close()

    foundHeader = False

    for line in infoList:
        l = line.strip()

        if 'Image Creator Information' in l:
            foundHeader = True

    infoFile = open( infoLoc, 'a' )
    if not foundHeader:
        infoFile.write('Image Creator Information\n')

    infoFile.write( '%s %d %d %d %d\n' % \
        ( pName, centers[0,0], centers[0,1], centers[1,0], centers[1,1] ))
    infoFile.close()

# End write to info file





# create dots on img 
def createDotImg( g1P, g2P, nRows, nCols, g1r, g2r, rConst ):

    img = np.zeros(( nRows, nCols ))

    g1rMax = np.amax( g1r )
    g2rMax = np.amax( g2r )

    for i,point in enumerate(g1P):
        x, y, z, r = point
        px = int(x)
        py = int(y)
        if px > 0 and px < nCols and py > 0 and py < nRows:
            img[nRows-py,px] += np.exp( -rConst * g1r[i] / g1rMax )  


    for i,point in enumerate(g2P):
        x, y, z, r = point
        px = int(x)
        py = int(y)
        if px > 0 and px < nCols and py > 0 and py < nRows:
            img[nRows-py,px] += np.exp( -rConst * g2r[i] / g2rMax )  

    return np.uint8(img*255)

# End create dot image

def simpleWriteImg( iLoc, g1P, g2P, nRows, nCols ):

    img = np.uint8( np.zeros(( nRows, nCols )))

    for x,y,z,r in g1P:
        px = int(x)
        py = int(y)
        if px > 0 and px < nCols and py > 0 and py < nRows:
            img[nRows-py,px] = 255

    for x,y,z,r in g2P:
        px = int(x)
        py = int(y)
        if px > 0 and px < nCols and py > 0 and py < nRows:
            img[nRows-py,px] = 255

    cv2.imwrite( iLoc, img )

# End simple write

# This is to shift the points so the galaxies are horizontal
def shiftPoints( g1P, g2P, gC, nRows, nCols ):

    if printAll:
        print('Shifing points to fit on image')

    # Calculate pixel points I want the galaxy centers to land on
    toC = np.zeros((2,2))
    toC[0,:] = [ nCols/3, nRows/2 ]
    toC[1,:] = [ 2*nCols/3, nRows/2 ]

    # Calculate amount needed to rotate points to land galaxy centers horizontal
    theta = - np.arctan( ( gC[1,1] - gC[0,1] ) / ( gC[1,0] - gC[0,0] ) ) + np.pi

    # Calculate scale to change galaxy center distance to pixel center distance
    scale = np.abs( toC[1,0] - toC[0,0] ) / np.sqrt( gC[1,0]**2 + gC[1,1]**2 )


    # Build rotation matrix
    rotMat = np.zeros((2,2))
    rotMat[0,:] = [ np.cos( theta ) , -np.sin( theta ) ]
    rotMat[1,:] = [ np.sin( theta ) ,  np.cos( theta ) ]


    # Only rotate xy plane
    gC[:,0:2] = np.transpose(np.matmul(rotMat, np.transpose(gC[:,0:2])))
    g1P[:,0:2] = np.transpose(np.matmul(rotMat, np.transpose(g1P[:,0:2])))
    g2P[:,0:2] = np.transpose(np.matmul(rotMat, np.transpose(g2P[:,0:2])))

   #scale up points, leave radial distance untouched
    gC[:,0:3] = scale*gC[:,0:3]
    g1P[:,0:3] = scale*g1P[:,0:3]
    g2P[:,0:3] = scale*g2P[:,0:3]

    # Shift centers up to desired point
    gC[:,0] = gC[:,0] + toC[0,0]
    gC[:,1] = gC[:,1] + toC[0,1]
    g1P[:,0] = g1P[:,0] + toC[0,0]
    g1P[:,1] = g1P[:,1] + toC[0,1]
    g2P[:,0] = g2P[:,0] + toC[0,0]
    g2P[:,1] = g2P[:,1] + toC[0,1]

    return g1P, g2P, gC

# end shift Points


def readParamFile( paramLoc ):

    if printAll:
        print('Reading image parameter file %s'%paramLoc)

    try:
        pFile = open( paramLoc, 'r')
        pList = list( pFile )
        pFile.close()
    except:
        print('Failed to read iamge parameter file %s' % paramLoc)
        exit(-1)

    pName = ''
    gSize = ''
    gWeight = ''
    rConts = ''
    normVal = ''
    nRows = ''
    nCols = ''

    for l in pList:
        l = l.strip()
        #print(l)

        if len(l) == 0:
            continue
        elif l[0] == '#':
            continue

        if 'parameter_name' in l:
            if printAll:
                print('Found %s'%l)
            pName = l.split()[1]
        
        elif 'gaussian_size' in l:
            if printAll:
                print('Found %s'%l)
            gSize = int(l.split()[1])

        elif 'gaussian_weight' in l:
            if printAll:
                print('Found %s'%l)
            gWeight = float(l.split()[1])

        elif 'radial_constant' in l:
            if printAll:
                print('Found %s'%l)
            rConst = float(l.split()[1])

        elif 'norm_value' in l:
            if printAll:
                print('Found %s'%l)
            normVal = float(l.split()[1])

        elif 'image_rows' in l:
            if printAll:
                print('Found %s'%l)
            nRows = int(l.split()[1])

        elif 'image_cols' in l:
            if printAll:
                print('Found %s'%l)
            nCols = int(l.split()[1])


    endEarly = False
    if printAll:
        print('%s,%f,%f,%f,%f,%d,%d'%(pName, gSize, gWeight, rConst, normVal, nRows, nCols) )

    try:
        return pName, gSize, gWeight, rConst, normVal,nRows, nCols 
    except:
        print("Failed to retrieve all parameters from file")
        exit(-1)

# End read param file

def readPartFile( pLoc ):
    
    if printAll:
        print('Reading particle file %s'%pLoc)

    try:
        pFile = open( pLoc, 'r' )
        pList = list( pFile )
        pFile.close()
    except:
        print('Failed to open particle file %s' % pLoc)
        exit(-1)
    
    nPart = int( (len( pList ) - 1 ) /2 )

    # End last line for galaxy centers
    cLine = pList[-1].strip()
    cVals = cLine.split()

    pCenters = np.zeros((2,3))

    g1Points = np.zeros((nPart,4))
    g2Points = np.zeros((nPart,4))

    # Read points
    for i in range(3):
        pCenters[1,i] = float(cVals[i])  # Error.  

    for i in range( nPart ):

        p1 = pList[i].strip().split()
        p2 = pList[i+nPart].strip().split()

        r1 = 0.0
        r2 = 0.0

        for j in range(3):

                g1Points[i,j] = float( p1[j] )
                g2Points[i,j] = float( p2[j] )

                r1 += g1Points[i,j]**2
                r2 += g2Points[i,j]**2

        g1Points[i,3] = np.sqrt( r1 )
        g2Points[i,3] = np.sqrt( r2 )


    return g1Points,g2Points,pCenters
    
# end read particle file    




def readArg():

    global printAll, maekMask, imageLoc, overwrite, runDir
    global infoLoc, paramLoc, partLoc1, partLoc2, paramInfo
    global keepZip

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-mask':
            makeMask = True

        elif arg == '-overwrite':
            overwrite = True

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir = runDir + '/'
            readRunDir(runDir)

        elif arg == '-paramLoc':
            paramLoc = argList[i+1]

        elif arg == '-imageLoc':
            imageLoc = argList[i+1]


    # Check if input arguments were valid
    endEarly = False
    
    if runDir == '':
        print('No run directory given')
    elif not path.exists(runDir):
        print('Run directory \'%s\' not found')
        endEarly = True

    if not path.isfile(partLoc1) or not path.isfile(partLoc2):
        print('Could not find particle files')
        endEarly = True

    if paramLoc == '' and paramInfo == '':
        print('Please specify image parameter location or information')
        endEarly = True

    if not path.isfile(paramLoc):
        print('Parameter file not found at: %s' % paramLoc)

# End reading command line arguments

def readRunDir( runDir ):
    global partLoc1, partLoc2, infoLoc

    zipFile1 = ''
    zipFile2 = ''

    try:
        dirList = listdir( runDir)

    except:
        print("Run directory '%s' not found" % runDir)
        return

    else:
        
        for f in dirList:
            fPath = runDir + f

            if 'info' in f:
                infoLoc = fPath

            elif '.000' in f:
                partLoc1 = fPath

            elif '.101' in f:
                partLoc2 = fPath

            elif '000.zip' in f:
                zipFile1 = fPath

            elif '101.zip' in f:
                zipFile2 = fPath
        
        if partLoc1 == '' and zipFile1 != '':
            unzip = 'unzip -d %s -o %s' % (runDir, zipFile1)
            system(unzip)

        if partLoc2 == '' and zipFile2 != '':
            unzip = 'unzip -d %s -o %s' % (runDir, zipFile2)
            system(unzip)

        '''

            lDir = listdir(runDir)
            sDir = ''
            for f in lDir:
                if '588' in f:
                    sDir = runDir + f

            rDir = sDir + '/' + listdir(sDir)[0]
            lDir = listdir(rDir)

            p1Loc = ''
            p2Loc = ''
            for f in lDir:
                if '.i.' in f:
                    p1Loc = rDir + '/' + f
                if '.f.' in f:
                    p2Loc = rDir + '/' + f

            mvCmd1 = 'mv %s %sa.000' % (p1Loc, runDir)
            mvCmd2 = 'mv %s %sa.101' % (p2Loc, runDir)

            system(mvCmd1)
            system(mvCmd2)
        '''


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
# end read argument file

# Run main after declaring functions
main()
