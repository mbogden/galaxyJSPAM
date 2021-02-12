'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    07 Jan 2020
Description:    Image creator for use in model to score pipeline. Currently in developement.

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
from numba import jit

# Global input variables
printAll = False
writeDotImg = False
overWriteImg = False

runDir = ''
paramLoc = ''
nPart = 100000

def image_creator_v4_pl(argList, imgParam = None, ignore=False):

    # Read input arguments
    if imgParam == None:
        endEarly = readArg( argList )
    else:
        endEarly = readArg( argList, ignore=True )

    if printAll:
        print('runDir       : %s' % runDir)
        print('paramLoc     : %s' % paramLoc)
        print('nPart        : %d' % nPart) 

    if endEarly and not ignore:
        print("Exiting...")
        return False

    runData = runDataClass(runDir, nPart)
    runGood = runData.checkRun()
    
    if not runGood:
        print("Run failed intialization: %s" % runDir)
        return False

    # If this is being used by another code
    if imgParam == None:
        imgParam = imageParameterClass_v3(paramLoc)


    runData.updateImg( imgParam )
    
    if not overWriteImg:
        imgExist = runData.checkImg()

        if imgExist:
            if printAll: print("Run images already created")
            return True

    # Read in particles
    runData.unZipPts()
    pts = readParticleFiles( runData.pts1Loc, runData.pts2Loc )
    runData.rmPts()

    Image_Run_Pipeline( runData, imgParam, pts )

# end image_creator_v


def Image_From_Files( paramLoc, pts1Loc, pts2Loc, imgLoc=None, circles=False ):
    
    # Get image parameters from file
    imgParam = imageParameterClass_v3( paramLoc )

    # Read particle files
    pts = readParticleFiles( pts1Loc, pts2Loc )

    # Shift particle so they align with image coordinates
    pts.g1fPart, pts.g2fPart, pts.fCenters2 = shiftPoints_v3( pts.g1fPart, 
            pts.g2fPart, pts.fCenters, imgParam.gCenter, imgParam.nRow )

    # Create image array
    modelImg = Image_From_Data( pts.g1fPart, pts.g2fPart, pts.ir1, pts.ir2, imgParam )

    # Add circles if desired
    if circles:
        modelImg = addCircles( modelImg, imgParam )

    # Save if specified
    if imgLoc != None:
        cv2.imwrite( imgLoc, modelImg )

    # Return image
    return modelImg

# End create image from files

# Make images if they within the automated pipeline
def Image_Run_Pipeline( runData, imgParam, pts ):

    # Create and save model image

    pts.g1fPart, pts.g2fPart, pts.fCenters2 = shiftPoints_v3( pts.g1fPart, 
            pts.g2fPart, pts.fCenters, imgParam.gCenter, imgParam.nRow)

    modelImg = Image_From_Data( pts.g1fPart, pts.g2fPart, pts.ir1, pts.ir2, imgParam )
    cv2.imwrite( runData.modelLoc, modelImg )

    # Create and save unperterbed image from initial points moved to final location
    dC = pts.fCenters - pts.iCenters
    pts.g2iPart[:,0:2] += dC[1,0:2]

    pts.g1iPart, pts.g2iPart, pts.iCenters2 = shiftPoints_v3( pts.g1iPart, pts.g2iPart, pts.fCenters, imgParam.gCenter, imgParam.nRow)

    initImg = Image_From_Data( pts.g1iPart, pts.g2iPart, pts.ir1, pts.ir2, imgParam )
    cv2.imwrite( runData.initLoc, initImg )

    return True

# End making images

def Image_From_Data( g1Pts, g2Pts, ir1, ir2, imgParam ):

    imgGal1 = addGalaxy( g1Pts, ir1, imgParam, imgParam.rConst1 )
    imgGal1 = cv2.GaussianBlur( imgGal1, (imgParam.gSize, imgParam.gSize), imgParam.gWeight )

    imgGal2 = addGalaxy( g2Pts, ir2, imgParam, imgParam.rConst2 )
    imgGal2 = cv2.GaussianBlur( imgGal2, (imgParam.gSize, imgParam.gSize), imgParam.gWeight )

    b1 = np.sum( imgGal1 )
    b2 = np.sum( imgGal2 )

    #print( imgParam.g1Lum, imgParam.g2Lum, b2, b1 )
    bScale = ( imgParam.g1Lum / imgParam.g2Lum ) * ( b2 / b1 )

    if bScale < 1:
        imgGal1 *= bScale
    else:
        imgGal2 *= ( 1 / bScale )

    finalImg = imgGal1 + imgGal2

    #finalImg = cv2.normalize( finalImg, np.zeros( finalImg.shape ), 0, 255, cv2.NORM_MINMAX)

    finalImg = normImg_v1( finalImg, imgParam.nVal )
    
    return finalImg

# end create Img

#@jit( nopython = True )
def shiftPoints_v3( g1pts, g2pts, ptsCenters, imgCenters, imgHeight, printTR=False ):
#def shiftPoints_v3( g1pts, g2pts, ptsCenters, imgParam, printTR=False ):
    
    #imgCenters = imgParam.gCenter
    #imgHeight = imgParam.nRow
    
    # x,y Cartesian coordinates for galaxy centers of particles
    g1_fc = ptsCenters[0,0:2]
    g2_fc = ptsCenters[1,0:2]
    
    # galaxy centers in pts format
    gc = np.ones((3,2))
    gc[0:2,0] = g1_fc
    gc[0:2,1] = g2_fc  

    # x, y cartesian coordinates for galaxy centers on img
    def ImgCartConv( x, y, mat_height ):    
        return np.array( [x, mat_height - y] )
    
    g1_tc = ImgCartConv( imgCenters[0,0], imgCenters[1,0], imgHeight)
    g2_tc = ImgCartConv( imgCenters[0,1], imgCenters[1,1], imgHeight)      
    
    # Find rotation angle between vectors.
    
    # Vector for centers
    fv = g2_fc - g1_fc  # from_vector: particle centers
    tv = g2_tc - g1_tc  # to_vector: image centers
    
    # signed angles vectors with (x=1,y=0)
    fv_angle = np.arctan2( fv[1], fv[0] )
    tv_angle = np.arctan2( tv[1], tv[0] )
    
    # signed angle from v1 to v2
    #theta = tv_angle - fv_angle
    theta = tv_angle - fv_angle
    
    # Create Rotation Matrix from theta
    rotMat = np.eye((3))
    rotMat[0,0:2] = [ np.cos( theta ) , -np.sin( theta ) ]
    rotMat[1,0:2] = [ np.sin( theta ) ,  np.cos( theta ) ]
    
    # Create Scaling Matrix    
    fl = np.sqrt( fv[0]**2 + fv[1]**2 ) # from center length
    tl = np.sqrt( tv[0]**2 + tv[1]**2 ) # to center length
    scale = tl/fl    
    
    scaleMat = np.eye(3)
    scaleMat[0,0] = scale # x-scale
    scaleMat[1,1] = scale # y-scale
    
    # Create Translation Matrix to overlap centers
    t1Mat = np.eye(3)
    t1Mat[0,2] = g1_tc[0]
    t1Mat[1,2] = g1_tc[1]
    
    
    # Create single adjustment matrix for computational effeciency        
    adjMat = np.copy(rotMat)    # rotate particles
    adjMat = np.dot( scaleMat, adjMat )  # Scale up values
    adjMat = np.dot( t1Mat, adjMat )  # translate to overlap
    

    if printTR:

        print('FROM:')
        print(gc[0:2,:])

        print("To:")
        print(imgCenters)  

        # Rotate points to match image
        print("\nRotate")
        adjMat2 = np.copy(rotMat)
        tMat = np.dot( rotMat, gc )
        print( np.dot( adjMat, gc ) )
        print( tMat )  

        # Scale points to match image
        adjMat2 = np.dot( scaleMat, adjMat )
        print("\nScale")
        tMat = np.dot( scaleMat, tMat )
        print( np.dot( adjMat, gc ) )
        print( tMat )

        # translate points to desired centers in cartesian
        adjMat2 = np.dot( t1Mat, adjMat )    
        print("\nTranslate to final centers cart")
        tMat = np.dot( t1Mat, tMat )
        print( np.dot( adjMat, gc ) )
        
        print("\nShould match")
        print( tMat )
        print( imgCenters )

        
    # APPLYING MATRIX TO POINTS
    # Transpose points because I (Matthew Ogden)
    # view "lists" of points as vertical arrays,
    # not horizontal as most matrix operations assume.

    n = g1pts.shape[0]

    # Shift center points
    gc = np.transpose( np.dot(adjMat, gc) )    

    # One's are required in 3rd dimension for proper translation
    tPts = np.ones((3,n))

    # Shift galaxy 1
    tPts[0:2,:] = g1pts[:,0:2].T    
    g1pts[:,0:2] = np.dot( adjMat, tPts )[0:2,:].T

    # Shift galaxy 2
    tPts[0:2,:] = g2pts[:,0:2].T
    g2pts[:,0:2] = np.dot( adjMat, tPts )[0:2,:].T    
    
    return g1pts, g2pts, gc

# End shifting points....  I really hope this is the last time I have to write this function


# Define struct-like class to hold arbitrary information
class particleClass:
    def __init__( self ):
        self.stuff = 0


def readParticleFiles( pts1Loc, pts2Loc ):
    
    pts = particleClass()
    
    pts.g1iPart, pts.g2iPart, pts.iCenters = readPartFile( pts1Loc )
    pts.g1fPart, pts.g2fPart, pts.fCenters = readPartFile( pts2Loc )
    
    pts.ir1 = pts.g1iPart[:,3]
    pts.ir2 = pts.g2iPart[:,3]
    
    return pts
# End simple read particles


def readPartFile( pLoc, printF=False ):
    
    if printF:
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

                # reading from file
                g1Points[i,j] = float( p1[j] )
                g2Points[i,j] = float( p2[j] )

                # calculating radius from center
                r1 += g1Points[i,j]**2
                r2 += g2Points[i,j]**2

        # finish radius calculation
        g1Points[i,3] = np.sqrt( r1 )
        g2Points[i,3] = np.sqrt( r2 )

    return g1Points, g2Points, pCenters
    
# end read particle file


# Define class and process if using run directory
class runDataClass:
    def __init__( self, runDir, nPart ):
        self.runDir = runDir
        self.nPart = nPart
        self.ptsDir = runDir + 'particle_files/'
        self.ptsZip = self.ptsDir + '%d_pts.zip' % nPart
        self.imgDir = runDir + 'model_images/'
        self.miscDir = runDir + 'misc_images/'
        self.infoLoc = runDir + 'info.txt'

    def checkRun( self ):
           
        if not path.exists(self.ptsDir):
            print("Couldn't find Folder: %s" % self.ptsDir)
            print(listdir(self.runDir))
            return False
           
        if not path.exists(self.imgDir):
            print("Couldn't find Folder: %s" % self.imgDir)
            print(listdir(self.runDir))
            return False
           
        if not path.exists(self.miscDir):
            print("Couldn't find Folder: %s" % self.miscDir)
            print(listdir(self.runDir))
            return False
           
        if not path.exists(self.ptsZip):
            print("Couldn't find particle file: %s" % self.ptsZip)
            print(listdir(self.ptsDir))
            return False
           
        if not path.exists(self.infoLoc):
            print("Couldn't find information file: %s" % self.infoLoc)
            print(listdir(self.runDir))
            return False

        return True
    
    def updateImg( self, imgParam ):

        self.modelLoc = self.imgDir + '%s_model.png' % imgParam.name 
        self.initLoc = self.imgDir + '%s_init.png' % imgParam.name 

    def checkImg( self ):

        # Check if images already exist
        if path.exists( self.modelLoc ):
            return True
        elif path.exists( self.initLoc ):
            return True
        else:
            return False

    
    def unZipPts( self ):

        unzipCmd = "unzip -qq -j -o %s -d %s" % ( self.ptsZip, self.ptsDir )
        system(unzipCmd)
        
        self.pts1Loc = self.ptsDir + "%d_pts.000" % self.nPart
        self.pts2Loc = self.ptsDir + "%d_pts.101" % self.nPart

        if not path.exists(self.pts1Loc) or not path.exists(self.pts2Loc):
            print("Can't find particle files after unzipping")
            print('\t', listdir( self.ptsDir ) )
            return False

        return True
    # End Unzipping Pts 

    def rmPts( self ):
        system('rm %s %s' % ( self.pts1Loc, self.pts2Loc ) ) 
    # end removing points
# End run class


# add Center circles
def addCircles(img, imgParam):
    g1c = ( int(imgParam.gCenter[0,0]) , int(imgParam.gCenter[1,0]) ) 
    g2c = ( int(imgParam.gCenter[0,1]) , int(imgParam.gCenter[1,1]) ) 
    cv2.circle( img, g1c, 10, (255, 255, 255), 2 ) 
    cv2.circle( img, g2c, 10, (255, 255, 255), 2 )
    return img


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



def addGalaxy( pts, ir, imgParam, rConst ):

    img = np.zeros(( imgParam.nRow, imgParam.nCol ))
    rMax = np.amax( pts[:,3] )

    for i, pt in enumerate(pts):
        x, y, z, r = pt
        x = int(x)
        y = int(y)
        
        if  x > 0 and x < imgParam.nCol \
        and y > 0 and y < imgParam.nRow:
            try:
                img[imgParam.nRow - y, x] += np.exp( -rConst * ir[i] / rMax )
            except:
                continue

    return img


# Define image parameter class
class imageParameterClass_v3:

    def __init__(self, pInLoc):

        self.gCenter    = np.zeros((2,2))   # [[ x1, x2 ] 
        self.comment    = 'blank comment'
        self.paramLoc   = pInLoc

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

            elif pL[0] == 'galaxy_1_luminosity':
                self.g1Lum = float(pL[1])

            elif pL[0] == 'galaxy_2_luminosity':
                self.g2Lum = float(pL[1])

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
        printList.append(' Galaxy 1 Luminosity     : %f' % ( self.g1Lum ))
        printList.append(' Galaxy 2 Luminosity     : %f' % ( self.g2Lum ))

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
            pFile.write('galaxy_1_luminosity %f' % ( self.g1Lum ))
            pFile.write('galaxy_2_luminosity %f' % ( self.g2Lum ))
            pFile.close()

    # end write param

    def checkParam():
        try:
            pV = self.printVal() # Also doubles as a check if all information needed is initialized
        except:
            print("Failed to read all Image parameters from file: %s" % paramLoc) 
            return False
        
        if self.version != 3:
            print("Incorrect image parameter verion:")
            print("Expecting: version 3")
            print("Found:     version %d" % self.version)
            return False

        # Return good if nothing bad found
        return True       

# End parameter class


def readArg( argList, ignore=False ):

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

    if paramLoc == '' and not ignore:
        print("No image parameter file given.")
        print("Example: \tpython image_creator.py -paramLoc path/to/param.txt")
        endEarly = True

    elif not path.exists(paramLoc) and not ignore:
        print("Image parameter file not found: %s" % paramLoc)
        endEarly = True
    

    return endEarly

# End reading command line arguments


# End simple read file

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

# End simple read file function

def test():
    print("In image creator version 4. 2020 Feb 3.")

# Run main after declaring functions
if __name__ == "__main__":
    argList = argv
    image_creator_v4_pl(argList)


