'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    11 Oct 2019
Description:    Image creator
python version 3, currently the version in developement.

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
writeDotImg = False
overWriteImg = False

runDir = ''
paramLoc = ''
nPart = 100000


def image_creator_pl_v1(argList):

    # Read input arguments
    endEarly = readArg( argList )

    if printAll:
        print('runDir       : %s' % runDir)
        print('paramLoc     : %s' % paramLoc)
        print('nPart        : %d' % nPart) 

    if endEarly:
        print("Exiting...")
        return False

    endEarly, runData = getRunData()

    if endEarly:
        print("Exiting...")
        return False

    # Read particle files
    g1iPart, g2iPart, iCenters = readPartFile( runData.pts1Loc )
    g1fPart, g2fPart, fCenters = readPartFile( runData.pts2Loc )
    ir1 = g1iPart[:,3]
    ir2 = g2iPart[:,3]
    
    # Remove uncompressed particles files after being read
    cleanPts = True
    if cleanPts:
        if printAll: print('rm %s %s' % ( runData.pts1Loc, runData.pts2Loc ) ) 
        system('rm %s %s' % ( runData.pts1Loc, runData.pts2Loc ) ) 


    # Create and save model image
    g1fPart, g2fPart, fCenters2 = shiftPoints( g1fPart, g2fPart, fCenters, runData.imgParam.nRow, runData.imgParam.nCol )

    modelImg = createImg( g1fPart, g2fPart, ir1, ir2, runData.bRatio, runData.imgParam )
    cv2.imwrite( runData.modelLoc, modelImg )


    # Create and save unperterbed image from initial points moved to final location
    dC = fCenters - iCenters
    g2iPart[:,0:2] += dC[1,0:2]

    g1iPart, g2iPart, iCenters2 = shiftPoints( g1iPart, g2iPart, fCenters, runData.imgParam.nRow, runData.imgParam.nCol )

    initImg = createImg( g1iPart, g2iPart, ir1, ir2, runData.bRatio, runData.imgParam )
    cv2.imwrite( runData.initLoc, initImg )

# end image_creator_v3

# Prepare directory and information
def getRunData():

    runData = runDataClass()

    # Reading from globals runDir, paramDir
    if printAll: print('prepping directory')

    # Check if particle files and information file is present
    endEarly, runData = readRunDir(runData)
    if endEarly: return endEarly

    runData.imgParam = imageParameterClass_v3(paramLoc)

    try:
        pV = runData.imgParam.printVal() # Also doubles as a check if all information needed is initialized
    except:
        print("Failed to read all Image parameters from file: %s" % paramLoc) 
        endEarly = True
        return endEarly, runData
    
    if printAll:
        print("Printing Image Parameter Info")
        for l in pV: print('\t',l)

    if runData.imgParam.version != 3:
        print("Incorrect image parameter verion:")
        print("Expecting: version 3")
        print("Found:     version %d" % runData.imgParam.version)
        print("Exiting...")
        endEarly = True
        return endEarly, runData
    
    runData.modelLoc = runData.imgDir + '%s_model_2.png' % runData.imgParam.name 
    runData.initLoc = runData.imgDir + '%s_init.png' % runData.imgParam.name 

    if path.exists( runData.modelLoc ) \
            and path.exists( runData.initLoc ) \
            and not overWriteImg:
                print("Model and unperturbed image already exist")
                endEarly = True
                return endEarly, runData

    # Get luminosity ratio
    g1Lum, g2Lum = getLuminosity( runData.infoLoc )
    g1Lum = 1.0
    g2Lum = 25.0
    if printAll: print( 'Lumisoties: %f %f' % ( g1Lum, g2Lum ) )
    if g1Lum == 0.0 or g2Lum == 0.0:
        endEarly = True
        return endEarly, runData

    runData.bRatio = g1Lum/g2Lum

    return endEarly, runData


# Define struct-like class to hold information
class runDataClass:
    def __init__( self ):
        # stuff
        self.stuff = 0


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

        elif 'secondary_luminosity' in l: 
            g2Lum = float( l.split()[1].strip() )

    # 

    return g1Lum, g2Lum

# End get Luminosity


def readRunDir(runData):

    if printAll: print("\tRun Directory: %s" % runDir )

    runData.ptsDir = runDir + 'particle_files/'
    runData.imgDir = runDir + 'model_images/'
    runData.miscDir = runDir + 'misc_images/'
    runData.infoLoc = runDir + 'info.txt'

    if not path.exists(runData.ptsDir) \
            or not path.exists(runData.imgDir) \
            or not path.exists(runData.miscDir) \
            or not path.exists(runData.infoLoc):

        print("Not all folders and files found in run directory")
        return True, runData

    ptsZip = runData.ptsDir + '%d_pts.zip' % nPart

    if not path.exists(ptsZip):
        print("Particle zip file not found: %s" % ptsZip)
        return True, runData

    unzipCmd = "unzip -qq -o %s -d %s" % ( ptsZip, runData.ptsDir )
    system(unzipCmd)
    
    runData.pts1Loc = runData.ptsDir + "%d_pts.000" % nPart
    runData.pts2Loc = runData.ptsDir + "%d_pts.101" % nPart

    if not path.exists(runData.pts1Loc) or not path.exists(runData.pts2Loc):
        print("Can't find particle files after unzipping")
        print("\tparticle file 1: %s" % runData.pts1Loc)
        print("\tparticle file 2: %s" % runData.pts2Loc)
        return True, runData

    if printAll:
        print("\tinfoLoc: %s" % runData.infoLoc)
        print("\tpts1Loc: %s" % runData.pts1Loc)
        print("\tpts2Loc: %s" % runData.pts2Loc)

    return False, runData

# End Reading run Folder


def readArg( argList ):

    global printAll, overWriteImg, runDir, writeDotImg, nPart, paramLoc

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-overwrite':
            overWriteImg = True

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


def normImg_v1( img, nVal ):

    maxVal = np.max( img )
    normImg = (img/maxVal)**(1/nVal)
    return normImg*255
# End normImg


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

            elif pL[0] == 'galaxy_1_center':
                self.gCenter[0,0] = int(pL[1])
                self.gCenter[1,0] = int(pL[2])

            elif pL[0] == 'galaxy_2_center':
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

    # Not complete
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
    # end write param

# End parameter class

# Run main after declaring functions
if __name__ == "__main__":
    argList = argv
    image_creator_pl_v1( argList )


