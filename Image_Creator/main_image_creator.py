'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Updated:    16 Nov 2020
Description:    This is my main program for all things image creation of galaxies
'''

from os import path, listdir
from sys import path as sysPath
import numpy as np
import cv2

# For loading in Matt's general purpose python libraries
sysPath.append( path.abspath( "Support_Code/" ) )
sysPath.append( path.abspath( path.join( __file__ , "../../Support_Code/" ) ) )
import general_module as gm
import info_module as im

def test():
    print("IC: Hi!  You're in Matthew's main code for all things image creation.")
    
def main(arg):


    if arg.printAll:

        arg.printArg()
        gm.test()
        im.test()
        
    # end main print
    
    if arg.simple:
        if arg.printBase: 
            print("PT: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None:
        image_creator_run( arg )

    else:
        print("PT: Nothing selected!")
        print("PT: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")

# End main


def image_creator_run( arg, ):
    
    # extract variables
    rDir = arg.runDir
    printBase=arg.printBase
    printAll=arg.printAll
    overwrite = arg.get('overwrite',False)
    
    rInfo = arg.get('rInfo')
    scoreParams = arg.get('score_parameters',None)

    if printBase:
        print("IC: image_creator_run")
        #arg.printArg()
    
    if rInfo == None:
        rInfo = im.run_info_class( runDir=rDir, printBase = False, printAll=printAll )

    if printBase:
        print('IC: rInfo.status: ', rInfo.status )

    if rInfo.status == False:
        print('IC: WARNGING:\n\t - rInfo status not good. Exiting...' )
        return
    
    if scoreParams == None:
        print("IC: WARNING: Please provide score parameters")
        return
    
    elif printAll:
        print("IC: given parameters: %d"%len(scoreParams))
            
    # Loop through files
    n = len(scoreParams)
    for i,pKey in enumerate(scoreParams):
        if printAll: print( "IC: ",pKey )
                    
        sParam = scoreParams[pKey]
        createImage( rInfo, sParam, printAll = printAll, overwrite=overwrite, )
        
        if printBase:
            print( 'IM_LOOP: %4d / %4d' % (i,n), end='\r' )
        
        
def createImage( rInfo, sParam, overwrite=False, printAll = False, ):
    
    if printAll: 
        print("IC: Creating image:")
        im.tabprint('runId: %s'%rInfo.get('run_id'))
        im.tabprint('score: %s'%sParam.get('name'))
        
    # Add place to keep points and images in memory
    if rInfo.get('img',None) == None:
        rInfo.img = {}
    if rInfo.get('pts',None) == None:
        rInfo.pts = {}
    
    imgName = sParam['imgArg'].get('name',None)
    simName = sParam['simArg'].get('name',None)

    # If image is already created, return
    imgMade = rInfo.findImgLoc( imgName, )
    if imgMade != None:
        if printAll: print("IC: Image '%s' already made"%imgName)
        if not overwrite: 
            img = gm.readImg( imgLoc )
            return img
    
    # Check if files are loaded
    pts = rInfo.pts.get(simName,None)
    img = rInfo.img.get(imgName,None)
        
    # Image location once made
    imgLoc = rInfo.findImgLoc( imgName, newImg = True )
    
    # Load points if not found
    if ( pts == None and img == None ) \
    or ( pts == None and img != None and overwrite ):
        
        if printAll: im.tabprint("Loading points from file")
        ptsZipLoc = rInfo.findPtsLoc( ptsName = simName )
        if ptsZipLoc == None:
            print("IC: WARNING: zipped points not found.")
            return None
            
        # Save pts in case needed for later
        rInfo.pts[simName] = particle_class( tmpDir = rInfo.tmpDir, zipLoc = ptsZipLoc, )  
        pts = rInfo.pts[simName]
    
    if overwrite or img == None:
        if printAll: im.tabprint("Creating image from points")
            
        imgArg = sParam['imgArg']
        img = pts2image( pts, imgArg )          
        # TODO Adjust brigtness on a galaxy

        # Apply gaussian blur
        img = blurImg( img, imgArg.get('blur',None))

        # Normalize image
        img = normImg( img, imgArg.get('normalization',None))

        if printAll: 
            im.tabprint("Saving image at: %s"%imgLoc)
        
    cv2.imwrite(imgLoc,img)
    return img
    
    
def pts2image( pts, imgArg, ):
    
    sg1f, sg2f, pts.sfCenters = shiftPoints( pts.g1f, pts.g2f, pts.fCenters, imgArg )
    
    # TO Initial image
    #pts.sg1i, pts.sg2i, pts.sfCenters = shiftPoints( pts.g1i, pts.g2i, pts.iCenters, imgArg )
       
    g1img = addGalaxy( sg1f, imgArg, 0 )
    g2img = addGalaxy( sg2f, imgArg, 1 )
    img = g1img + g2img
    
    return img


# Blur image
def blurImg( img, blurArg ):
    
    # default, do nothing
    if blurArg == None:
        return img
    
    bType = blurArg.get('type')
    
    if bType == 'gaussian_blur':
        size = blurArg.get('size')
        weight = blurArg.get('weight')
        img = cv2.GaussianBlur( img, (size,size), weight)
        return img
    
    else: 
        print("IC: blurImg WARNING: blur type not found %s:"%bType)
        
    return img
# End blur image


def normImg( img, normArg ):
    
    if normArg == None:
        normType = 'linear'
        
    else:
        normType = normArg.get('type')
    
    if normType == 'linear':
        img = cv2.normalize( img, np.zeros( img.shape ), 0, 255, \
                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img
    
    elif normType == 'type2':
        
        nVal = normArg.get('norm_value')
        maxVal = np.max( img )
        img = (img/maxVal)**(1/nVal)        
        img = cv2.normalize( img, np.zeros( img.shape ), 0, 255, \
                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return img
    
    else:
        print("IM: WARNING:  Normalization type not found")
        img = cv2.normalize( img, np.zeros( img.shape ), 0, 255, \
                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# End normImg   
        


# Define class for reading and hodling initial part files
class particle_class:
    
    status = False
    
    def __init__( self, zipLoc = None, tmpDir=None, printAll=False ):
                
        self.printAll = printAll
        
        # Unzip zip file
        pts1Loc, pts2Loc = self.openPtsZip( zipLoc, tmpDir )
        if pts1Loc == None or pts2Loc == None:
            if printAll: print("IM: WARNING: Can't find pts files in zip: %s",zipLoc)
            return
        
        # Read from txt files
        self.g1i, self.g2i, self.iCenters = self.readPartFile( pts1Loc )
        self.g1f, self.g2f, self.fCenters = self.readPartFile( pts2Loc )
        
        # Offset unperterbed points to overlap with final location
        #dC = self.fCenters - self.iCenters
        #self.og2i = np.copy(self.g2i)
        #self.og2i[:,0:2] += dC[1,0:2]
        
        if printAll:
            print(self.g1i.shape, self.g2i.shape, self.iCenters.shape)
            print(self.g1f.shape, self.g2f.shape, self.fCenters.shape)
            
    # End initializing particle files.
    
    def openPtsZip( self, zipLoc, tmpDir ):

        if not path.exists(tmpDir):
            from os import mkdir
            if self.printAll: print('IC: Making dir: %s'%tmpDir)
            mkdir(tmpDir)
        
        import zipfile
        with zipfile.ZipFile(zipLoc, 'r') as zip_ref:
            zip_ref.extractall(tmpDir)
        
        pts1Loc = None
        pts2Loc = None
        for f in listdir( tmpDir ):
            if '.000' in f:
                pts1Loc = tmpDir + f
            if '.101' in f:
                pts2Loc = tmpDir + f
        
        return pts1Loc,pts2Loc
    # End unziping pts file
    
    def readPartFile( self, pLoc, ):
        
        # Read file using numpy reading
        pts = np.genfromtxt( pLoc )
        
        # number of particle per galaxy
        nPart = int( (pts.shape[0]-1)/2 )
        
        pCenters = np.zeros((2,3))
        g1Points = np.zeros((nPart,3))
        g2Points = np.zeros((nPart,3))
        
        pCenters[1,:] = pts[-1,0:3]
        
        # Get x,y,z. leave room for r
        #print('SHAPE: ',pts[:nPart,0:3].shape, pts[nPart:2*nPart,0:3].shape)
        g1Points[:,:] = pts[:nPart,0:3]
        g2Points[:,:] = pts[nPart:2*nPart,0:3]
        
        g1Points[:,2] = np.sqrt( g1Points[:,0]**2 + g1Points[:,1]**2 + g1Points[:,2]**2 )
        g2Points[:,2] = np.sqrt( (pCenters[1,0] - g2Points[:,0])**2 \
                                + (pCenters[1,1] - g2Points[:,1])**2 \
                                + (pCenters[1,2] - g2Points[:,2])**2 )
                
        return g1Points, g2Points, pCenters
     
    # end read particle file


# Working only if pixel centers are on x axis
def shiftPoints( g1pts, g2pts, ptsCenters, imgArg ):
    
    imgHeight = imgArg['image_size'][0]
    imgCenters = np.zeros((2,2))
    imgCenters[:,0] = imgArg['pixel_centers']['1']
    imgCenters[:,1] = imgArg['pixel_centers']['2']
    
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
    #print(g1_tc,g2_tc)
    
    g1_tc = imgCenters[:,0]
    g2_tc = imgCenters[:,1]
    
    # Find rotation angle between vectors.
    
    # Vector for centers
    fv = g2_fc - g1_fc  # from_vector: particle centers
    tv = g2_tc - g1_tc  # to_vector: image centers
    
    # signed angles vectors with (x=1,y=0)
    fv_angle = np.arctan2( fv[1], fv[0] )
    tv_angle = np.arctan2( tv[1], tv[0] )
    
    # signed angle from v1 to v2
    theta = tv_angle - fv_angle
    theta = tv_angle - fv_angle + np.pi/2
    
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
    
    
    if False:

        print('[[ x1, x2],')
        print(' [ y1, y2]]')
        print('FROM:')
        print(gc[0:2,:])

        print("To:")
        print(imgCenters)  

        # Rotate points to match image
        print("\nRotate")
        adjMat2 = np.copy(rotMat)
        tMat = np.dot( rotMat, gc )
        print( np.dot( adjMat2, gc ) )
        print( tMat )

        # Scale points to match image
        adjMat2 = np.dot( scaleMat, adjMat2 )
        print("\nScale")
        tMat = np.dot( scaleMat, tMat )
        print( np.dot( adjMat2, gc ) )
        print( tMat )

        # translate points to desired centers in cartesian
        adjMat2 = np.dot( t1Mat, adjMat2 )    
        print("\nTranslate to final centers cart")
        tMat = np.dot( t1Mat, tMat )
        print( np.dot( adjMat2, gc ) )
        print( np.dot( adjMat, gc ) )
        print( tMat )
        
        print("\nShould match")
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
    sg1pts = np.zeros((n,3))
    sg2pts = np.zeros((n,3))

    # Shift galaxy 1
    tPts[0:2,:] = g1pts[:,0:2].T    
    sg1pts[:,0:2] = np.dot( adjMat, tPts )[0:2,:].T  
    sg1pts[:,2] = g1pts[:,2]

    # Shift galaxy 2
    tPts[0:2,:] = g2pts[:,0:2].T  
    sg2pts[:,0:2] = np.dot( adjMat, tPts )[0:2,:].T    
    sg2pts[:,2] = g2pts[:,2]
    
    return sg1pts, sg2pts, gc

# End shifting points....  I really hope this is the last time I have to write this function

def addGalaxy( ptSet, imgArg, gNum=None ):

    imgSize = (imgArg['image_size'][0],imgArg['image_size'][1])
    rConst = imgArg.get( 'radial_const', None )
    
    if rConst == None:
        rConst = None
        
    else:
        rConst = rConst[gNum]
        
    x = ptSet[:,0]
    y = ptSet[:,1]
    r = ptSet[:,2]
    rMax = np.amax(r)
    
    # For radial constant correction
    weights = np.ones(x.shape[0])
    if rConst != None:
        weights = np.exp( -rConst * r / rMax )
    
    # histogram 2D points onto image
    xedges = np.linspace(0,imgSize[1],imgSize[1]+1)
    yedges = np.linspace(0,imgSize[0],imgSize[0]+1)
    img, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), weights = weights)
    
    return img
# End adding galaxy to a image

# add Center circles
def addCircles(img, imgParam):
    
    im.pprint(imgParam)
    g1c = ( imgParam['pixel_centers']['1'][0] , imgParam['pixel_centers']['1'][1] ) 
    g2c = ( imgParam['pixel_centers']['2'][0] , imgParam['pixel_centers']['2'][1] )
    
    cimg = np.copy(img)
    cv2.circle( cimg, g1c, 7, (255, 255, 255), 2 ) 
    cv2.circle( cimg, g2c, 7, (255, 255, 255), 2 )
    return cimg



# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
