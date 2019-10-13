'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    11 Oct 2019
Description:    This is my python version 3, currently the version in developement.

ToDo:
    Different brightness in each galaxy
    Integrate with new pipeline layout.
    New image parameter file 
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
overwriteImage = False
saveInit = True
writeDotImg = False

runDir = ''
paramLoc = ''
nPart = 100000

def image_creator_v3(argList):

    # Read input arguments
    endEarly = nReadArg( argList )

    if printAll:
        print('runDir       : %s' % runDir)
        print('paramLoc     : %s' % paramLoc)
        print('nPart        : %d' % nPart) 

    if endEarly:
        print("Exiting...")
        return False

    imgParam = imageParameterClass_v3(paramLoc)
    pV = imgParam.printVal()
    if printAll:
        for l in pV: print(l)

    if imgParam.version != 3:
        print("Incorrect image parameter verion:")
        print("Expecting: version 3")
        print("Found:     version %d" % imgParam.version)
        print("Exiting...")
        return False

    endEarly, infoLoc, pts1Loc, pts2Loc = nReadRunDir()
    if endEarly:
        print("Exiting...")
        return False

    g1Lum, g2Lum = getLuminosity( infoLoc )
    if printAll: print( 'Lumisoties: %f %f' % ( g1Lum, g2Lum ) )
    if g1Lum == 0.0 or g2Lum == 0.0:
        endEarly = True
        print("Exiting...")
        return False

    bRatio = g1Lum/g2Lum

    # Read particle files
    g1iPart, g2iPart, iCenters = readPartFile( pts1Loc )
    g1fPart, g2fPart, fCenters = readPartFile( pts2Loc )
    ir1 = g1iPart[:,3]
    ir2 = g2iPart[:,3]
    
    # Remove uncompressed 
    cleanPts = True
    if cleanPts:
        if printAll: print('rm %s %s' % ( pts1Loc, pts2Loc ) ) 
        system('rm %s %s' % ( pts1Loc, pts2Loc ) ) 


    # Write model image

    g1fPart, g2fPart, fCenters2 = shiftPoints( g1fPart, g2fPart, fCenters, imgParam.nRow, imgParam.nCol )

    modelImg = createImg( g1fPart, g2fPart, ir1, ir2, bRatio, imgParam )
    imgLoc = runDir + 'model_images/%s_model.png' % imgParam.name 
    cv2.imwrite( imgLoc, modelImg )


    # Create unperterbed image from initial points moved to final location

    dC = fCenters - iCenters
    g2iPart[:,0:2] += dC[1,0:2]
    g1iPart, g2iPart, iCenters2 = shiftPoints( g1iPart, g2iPart, fCenters, imgParam.nRow, imgParam.nCol )

    initImg = createImg( g1iPart, g2iPart, ir1, ir2, bRatio, imgParam )
    initImageLoc = runDir + 'model_images/%s_init.png' % imgParam.name
    cv2.imwrite( initImageLoc, initImg )

# end image_creator_v3

def createImg( g1Pts, g2Pts, ir1, ir2, bRatio, imgParam ):

    imgGal1 = addGalaxy( g1Pts, ir1, imgParam, imgParam.rConst1 )
    imgGal1 = cv2.GaussianBlur( imgGal1, (imgParam.gSize, imgParam.gSize), imgParam.gWeight )

    imgGal2 = addGalaxy( g2Pts, ir2, imgParam, imgParam.rConst2 )
    imgGal2 = cv2.GaussianBlur( imgGal2, (imgParam.gSize, imgParam.gSize), imgParam.gWeight )

    b1 = np.sum( imgGal1 )
    b2 = np.sum( imgGal2 )

    bScale = bRatio * b2 / b1 

    if bScale < 1:
        imgGal1 *= bScale
    else:
        imgGal2 *= ( 1 / bScale )

    finalImg = imgGal1 + imgGal2

    #finalImg = cv2.normalize( finalImg, np.zeros( finalImg.shape ), 0, 255, cv2.NORM_MINMAX)

    finalImg = normImg_v1( finalImg, imgParam.nVal )
    
    return finalImg

# end create Img

def addGalaxy( pts, ir, imgParam, rConst ):

    img = np.zeros(( imgParam.nRow, imgParam.nCol ))
    rMax = np.amax( pts[:,3] )

    for i, pt in enumerate(pts):
        x, y, z, r = pt
        x = int(x)
        y = int(y)
        
        if  x > 0 and x < imgParam.nCol \
        and y > 0 and y < imgParam.nRow:
            img[imgParam.nRow - y, x] += np.exp( -rConst * ir[i] / rMax )

    return img


def getLuminosity( infoLoc ):
    
    if printAll: print("Getting gal brightnesses")

    infoFile = readFile( infoLoc )

    g1Lum = 0.0
    g2Lum = 0.0

    for l in infoFile:
        if 'primary_luminosity' in l: 
            g1Lum = float( l.split()[1].strip() )

        if 'secondary_luminosity' in l: 
            g2Lum = float( l.split()[1].strip() )

    return g1Lum, g2Lum


def nReadRunDir():

    if printAll: print("In run dir: %s" % runDir )

    ptsDir = runDir + 'particle_files/'
    imgDir = runDir + 'model_images/'
    miscDir = runDir + 'misc_images/'
    infoLoc = runDir + 'info.txt'

    if not path.exists(ptsDir) \
            or not path.exists(imgDir) \
            or not path.exists(miscDir) \
            or not path.exists(infoLoc):

        print("Not all folders and files found in run directory")
        return True, '', '', ''

    ptsZip = ptsDir + '%d_pts.zip' % nPart

    if not path.exists(ptsZip):
        print("Particle zip file not found: %s" % ptsZip)
        return True, '', '', ''

    unzipCmd = "unzip -qq -o %s -d %s" % ( ptsZip, ptsDir )
    system(unzipCmd)
    
    pts1Loc = ptsDir + "%d_pts.000" % nPart
    pts2Loc = ptsDir + "%d_pts.101" % nPart

    if not path.exists(pts1Loc) or not path.exists(pts2Loc):
        print("Can't find particle files after unzipping")
        print("\tparticle file 1: %s" % pts1Loc)
        print("\tparticle file 2: %s" % pts2Loc)
        return True, '', '', ''


    if printAll:
        print("\tinfoLoc: %s" % infoLoc)
        print("\tpts1Loc: %s" % pts1Loc)
        print("\tpts2Loc: %s" % pts2Loc)

    return False, infoLoc, pts1Loc, pts2Loc

# End Reading run Folder


def nReadArg( argList ):

    global printAll, overWrite, runDir, writeDotImg, nPart, paramLoc

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-overwrite':
            overwrite = True

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir = runDir + '/'

        elif arg == '-paramLoc':
            paramLoc = argList[i+1]

        elif arg == '-dotImg':
            writeDotImg = True

        elif arg == '-init':
            saveInit = True

        elif arg == '-nPart':
            nPart = argList[i+1]
            try:
                nPart = int( nPart)
            except:
                print("Number of particles not recognized as integer: %s" % nPart) 

    # Check if input arguments were valid
    endEarly = False
    
    if runDir == '':
        print('No run directory given')
        endEarly = True

    elif not path.exists(runDir):
        print('Run directory \'%s\' not found')
        endEarly = True

    if paramLoc == '':
        print("No image parameter file given.")
        endEarly = True

    elif not path.exists(paramLoc):
        print("Image parameter file not found: %s" % paramLoc)
        endEarly = True
    

    return endEarly

# End reading command line arguments



def image_creator_v2( inArg ):

    g1iPart, g2iPart, iCenters = readPartFile( partLoc1 )
    g1fPart, g2fPart, fCenters = readPartFile( partLoc2 )


    if imageLoc == '':
        imageLoc = runDir + '%s_model.png' % pName


    if printAll: print("Saving image at %s" % imageLoc)

    g1fPart, g2fPart, fCenters2 = shiftPoints( g1fPart, g2fPart, fCenters, nRows, nCols )

    if True:
        simpleWriteImg( runDir + 'simple.png', g1fPart, g2fPart, nRows, nCols )

    # Add particles to image
    dotImg = addParticles( g1fPart, g2fPart, nRows, nCols, g1iPart[:,3], g2iPart[:,3], rConst )

    if writeDotImg:
        dotNormImg = normImg_v1( dotImg, normVal )
        #dotNormImg = cv2.normalize( dotImg, np.zeros( dotImg.shape ), 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite( runDir + '%s_dot.png' % pName, dotNormImg )

    # Perform gaussian blur
    blurImg = cv2.GaussianBlur( dotImg, (gSize, gSize), gWeight )
    #blurImg = cv2.bilateralFilter( dotImg, int(gWeight), gSize, gSize )

    # Normalize brightness to image format
    if False:
        normImg = cv2.normalize( blurImg, np.zeros( blurImg.shape ), 0, 255, cv2.NORM_MINMAX)
    else:
        normImg = normImg_v1( blurImg, normVal )

    # Write image
    cv2.imwrite( imageLoc, normImg )
    writeParamInfo( infoLoc, pName, fCenters2 )

    if saveInit:

        print("Saving Shifted Initial Particle Image")
        initImageLoc = runDir + '%s_model_init.png' % pName
        initDiffLoc = runDir + '%s_model_init_diff.png' % pName

        # Shift galaxy 2 to where it would be in final image
        dC = fCenters - iCenters
        for i in range(g2iPart.shape[0]):
            g2iPart[i,0:2] += dC[1,0:2]

        g1iPart2, g2iPart2, iCenters2 = shiftPoints( g1iPart, g2iPart, fCenters, nRows, nCols )
        dotImg = addParticles( g1iPart2, g2iPart2, nRows, nCols, g1iPart[:,3], g2iPart[:,3], rConst )

        blurImg = cv2.GaussianBlur( dotImg, (gSize, gSize), gWeight )
        normImg2 = normImg_v1( blurImg, normVal )
        initDiffImg = np.abs( normImg2 - normImg )

        cv2.imwrite( initImageLoc, normImg2 )
        cv2.imwrite( initDiffLoc, initDiffImg )

    cleanUpDir()
    # End saving inital particles

# End main

def cleanUpDir():

    if path.exists(zipFile1):
        system("rm %s" % partLoc1)

    if path.exists(zipFile2):
        system("rm %s" % partLoc2)

    # delete unziped files


def normImg_v1( img, nVal ):

    maxVal = np.max( img )
    normImg = (img/maxVal)**(1/nVal)
    return normImg*255
# End normImg



def addParticles( g1P, g2P, nRows, nCols, g1r, g2r, rConst ):

    rawDotImg = np.zeros(( nRows, nCols ))

    g1rMax = np.amax( g1r )
    g2rMax = np.amax( g2r )

    for i,point in enumerate(g1P):
        x, y, z, r = point
        px = int(x)
        py = int(y)
        if px > 0 and px < nCols and py > 0 and py < nRows:
            rawDotImg[nRows-py,px] += np.exp( -rConst * g1r[i] / g1rMax )  


    for i,point in enumerate(g2P):
        x, y, z, r = point
        px = int(x)
        py = int(y)
        if px > 0 and px < nCols and py > 0 and py < nRows:
            rawDotImg[nRows-py,px] += np.exp( -rConst * g2r[i] / g2rMax )  

    return np.float32(rawDotImg)

# End adding points to image



def simpleWriteImg( iLoc, g1P, g2P, nRows, nCols ):

    img = np.uint8( np.zeros(( nRows, nCols )))

    c = 0
    for x,y,z,r in g1P:
        px = int(x)
        py = int(y)
        if c%10 == 0 and px > 0 and px < nCols and py > 0 and py < nRows:
            img[nRows-py,px] = 255
        c += 1

    for x,y,z,r in g2P:
        px = int(x)
        py = int(y)
        if c%10 == 0 and px > 0 and px < nCols and py > 0 and py < nRows:
            img[nRows-py,px] = 255
        c += 1

    cv2.imwrite( iLoc, img )

# End simple write

# This is to shift the points so the galaxies are horizontal
def shiftPoints( g1P, g2P, gC_in, nRows, nCols ):

    gC = np.copy(gC_in)

    # Calculate pixel points I want the galaxy centers to land on
    toC = np.zeros((2,2))
    toC[0,:] = [ nCols/3, nRows/2 ]
    toC[1,:] = [ 2*nCols/3, nRows/2 ]

    # Calculate amount needed to rotate points to land galaxy centers horizontal
    theta = - np.arctan( ( gC[1,1] - gC[0,1] ) / ( gC[1,0] - gC[0,0] ) ) 

    # Add pi if galaxy is on wrong side
    if gC[1,0] < 0:
        theta += np.pi


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

def readPartFile( pLoc ):
    
    if printAll:
        print('Reading particle file: %s'%pLoc)

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


# Define image parameter class
class imageParameterClass_v3:

    def __init__(self, pInLoc):

        self.status     = 'starting'
        self.gCenter    = np.zeros((2,2))   # [[ x1, x2 ] 
        self.comment    = 'blank comment'

        self.readParamFile(pInLoc)

    # end init

    def readParamFile(self, pInLoc):

        if not path.exists(pInLoc):
            print("Image parameter file not found: %s" % pInLoc)
            self.status = 'bad'
            return 

        try:
            paramFile = open(pInLoc,'r')

        except:
            print("Failed to open image parameter file: %s" % pInLoc)
            self.status = 'bad'
            return 

        else:
            pFile = list( paramFile )
            paramFile.close()


        for line in pFile:
            l = line.strip()
            if len(l) == 0:
                continue
            
            if l[0] == '#':
                self.comment = l
            
            pL = l.split(' ')

            if pL[0] == 'parameter_name':
                self.name = pL[1] 

            elif pL[0] == 'version':
                self.version = int(pL[1])   

            elif pL[0] == 'gaussian_size':
                self.gSize = int(pL[1])   

            elif pL[0] == 'gaussian_weight':
                self.gWeight = float(pL[1])   

            elif pL[0] == 'radial_constant1':
                self.rConst1 = float(pL[1])   

            elif pL[0] == 'radial_constant2':
                self.rConst2 = float(pL[1])   

            elif pL[0] == 'brightness_constant':
                self.bConst = float(pL[1]) 

            elif pL[0] == 'norm_value':
                self.nVal = float(pL[1]) 

            elif pL[0] == 'image_rows':
                self.nRow = int(pL[1]) 

            elif pL[0] == 'image_cols':
                self.nCol = int(pL[1]) 

            elif pL[0] == 'galaxy1_center':
                self.gCenter[0,0] = int(pL[1])
                self.gCenter[1,0] = int(pL[2])

            elif pL[0] == 'galaxy2_center':
                self.gCenter[0,1] = int(pL[1])
                self.gCenter[1,1] = int(pL[2])

    # end read param file

    def printVal(self):

        printList = []

        printList.append(' Name                    : %s' % self.name)
        printList.append(' Version                 : %s' % self.version)
        printList.append(' Comment                 : %s' % self.comment)
        printList.append(' Gaussian size           : %d' % self.gSize)
        printList.append(' Gaussian weight         : %f' % self.gWeight)
        printList.append(' Radial constant1        : %f' % self.rConst1)
        printList.append(' Radial constant2        : %f' % self.rConst2)
        printList.append(' Brightness constant     : %f' % self.bConst)
        printList.append(' Normalization constant  : %f' % self.nVal)
        printList.append(' Number of rows          : %d' % self.nRow)
        printList.append(' Number of columns       : %d' % self.nCol)
        printList.append(' Galaxy 1 center         : %d %d' % ( int(self.gCenter[0,0]), int(self.gCenter[0,1]) ))
        printList.append(' Galaxy 2 center         : %d %d' % ( int(self.gCenter[1,0]), int(self.gCenter[1,1]) ))

        return printList
    # end print

    def writeParam(self, saveLoc):
        try:
            pFile = open(saveLoc,'w')
        except:
            print('Failed to create: %s' % saveLoc)
        else:
            pFile.write('parameter_name %s\n' % self.name)
            pFile.write('# %s\n\n' % self.comment)
            pFile.write('gaussian_size %d\n' % self.gSize)
            pFile.write('gaussian_weight %f\n' % self.gWeight)
            pFile.write('radial_constant1 %f\n' % self.rConst1)
            pFile.write('radial_constant2 %f\n' % self.rConst2)
            pFile.write('brightness_constant %f\n' % self.bConst)
            pFile.write('norm_value %f\n' % self.nVal)
            pFile.write('image_rows %d\n' % self.nRow)
            pFile.write('image_cols %d\n' % self.nCol)
            pFile.write('galaxy_1_center %d %d\n' % ( int(self.gCenter[0,0]), int(self.gCenter[0,1]) ))
            pFile.write('galaxy_2_center %d %d\n' % ( int(self.gCenter[1,0]), int(self.gCenter[1,1]) ))
            pFile.close()


# End parameter class

# Run main after declaring functions
if __name__ == "__main__":
    #new_main()
    argList = argv
    image_creator_v3( argList )

