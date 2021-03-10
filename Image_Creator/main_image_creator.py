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
    
# For testing and developement
new_func = None
def set_new_func( inFunc ):
    global new_func
    new_func = inFunc
# End set new function

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


def main_ic_run( arg, ):
    
    # extract variables
    rDir = arg.runDir
    printBase=arg.printBase
    printAll=arg.printAll
    overWrite = arg.get('overWrite',False)
    
    rInfo = arg.get('rInfo')
    scoreParams = arg.get('scoreParams',None)

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
        create_image_from_parameters( rInfo, sParam, printAll = printAll, overwrite=overWrite, )
        
        if printBase:
            print( 'IM_LOOP: %4d / %4d' % (i,n), end='\r' )
            
# End main image creator run
        
def create_image_from_parameters( rInfo, sParam, overwrite=False, printAll = False, ):
    
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
    imgLoc = rInfo.findImgLoc( imgName, )
    if imgLoc != None:
        if printAll: print("IC: Image '%s' already made"%imgLoc)
        if not overwrite: 
            img = gm.readImg( imgLoc )
            rInfo.img[imgName] = img
            return img
    
    # Check if files are loaded into info file
    pts = rInfo.pts.get(simName,None)
    
    # Load points if not found
    if pts == None:
        
        if printAll: im.tabprint("Loading points from file")
        ptsZipLoc = rInfo.findPtsLoc( ptsName = simName )
        if ptsZipLoc == None:
            print("IC: WARNING: zipped points not found.")
            return None
            
        # Save pts in case needed for later
        pts = particle_class( tmpDir = rInfo.tmpDir, zipLoc = ptsZipLoc, )  
        rInfo.pts[simName] = pts        
    
    if printAll: im.tabprint("Creating image from points")

    imgArg = sParam['imgArg']
    img = pts2image( pts, imgArg )
    # TODO Adjust brigtness on a galaxy

    # Apply blur
    img = blurImg( img, imgArg )

    # Normalize image
    img = normImg( img, imgArg )

    if printAll: 
        im.tabprint("Saving image at: %s"%imgLoc)

    # Save image in case needed later
    rInfo.img[imgName] = img
        
    # Image location 
    imgLoc = rInfo.findImgLoc( imgName, newImg = True )
    cv2.imwrite(imgLoc,img)
    
    return img
    
    
def pts2image( pts, imgArg, ):
    
    sg1f, sg2f, pts.sfCenters = shiftPoints( pts.g1f, pts.g2f, pts.fCenters, imgArg )
    #sg1f, sg2f, pts.sfCenters = new_func( pts.g1f, pts.g2f, pts.fCenters, imgArg )
    
    # TO Initial image
    #pts.sg1i, pts.sg2i, pts.sfCenters = shiftPoints( pts.g1i, pts.g2i, pts.iCenters, imgArg )
       
    g1img = addGalaxy( sg1f, imgArg, 0 )
    g2img = addGalaxy( sg2f, imgArg, 1 )
    img = g1img + g2img
    
    return img


# Blur image
def blurImg( img, imgArg ):
    
    blurArg = imgArg.get('blur',None)
    
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


def normImg( img, imgArg ):
    
    # Get arguments for normalizing    
    normArg = imgArg.get('normalization',None)
    
    # If none give, assume linear
    if normArg == None:
        normType = 'linear'
        normArg = {'type':'linear'}
        
    else:
        normType = normArg.get('type')
    
    # If max brightness present, make all pixels brighter than max equal max  
    lMax = normArg.get('max_brightness',None)
    if lMax != None:      
        img[img>lMax] = lMax      
    
    # Set highest value to 255, lowest to 0
    if normType == 'linear':
        img = cv2.normalize( img, np.zeros( img.shape ), 0, 255, \
                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img    
    
    elif normType == 'type1':
        
        nVal = normArg.get('norm_constant')
        img = np.power( img, 1/nVal )
        img = cv2.normalize( img, np.zeros( img.shape ), 0, 255, \
                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return img
    
    # 
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
def shiftPoints( g1pts, g2pts, ptsCenters, imgArg, devPrint=False ):
    
    # Grab initial values
    imgHeight = imgArg['image_size']['height']
    
    # Final image (m) coordinates for galaxy centers
    g1_mf = np.zeros(2)
    g2_mf = np.zeros(2)
    g1_mf[0] = imgArg['galaxy_centers']['px']
    g1_mf[1] = imgArg['galaxy_centers']['py']
    g2_mf[0] = imgArg['galaxy_centers']['sx']
    g2_mf[1] = imgArg['galaxy_centers']['sy']
    
    # Initial Cartesian (c) coordinates for galaxy centers
    g1_ci = ptsCenters[0,0:2]
    g2_ci = ptsCenters[1,0:2]
    
    # galaxy centers in pts format
    gc = np.ones((3,2))
    gc[0:2,0] = g1_ci
    gc[0:2,1] = g2_ci

    # Converting y cartesian coordinates from img to cart and vice-versa
    def ImgCartConv( y, mat_height ):
        return np.array( mat_height - y )
    
    # Convert final img points to cart points
    g1_cf = np.copy(g1_mf)
    g2_cf = np.copy(g2_mf)
    g1_cf[1] = ImgCartConv( g1_mf[1], imgHeight)
    g2_cf[1] = ImgCartConv( g2_mf[1], imgHeight)

    # Find rotation angle between vectors.  
    iv = g2_ci - g1_ci  # initial_vector:
    fv = g2_cf - g1_cf  # final_vector:
    
    # angles of the vectors
    iv_angle = np.arctan2( iv[1], iv[0] )
    fv_angle = np.arctan2( fv[1], fv[0] )
    
    # angle between final and initial
    theta = fv_angle - iv_angle
    
    # make positive for my sake.
    if theta < 0:
        theta += 2*np.pi
    
    # Create Rotation Matrix from theta
    rotMat = np.eye((3))
    rotMat[0,0:2] = [ np.cos( theta ) , -np.sin( theta ) ]
    rotMat[1,0:2] = [ np.sin( theta ) ,  np.cos( theta ) ]
    
    # Create Scaling Matrix
    il = np.sqrt( iv[0]**2 + iv[1]**2 ) # initial center length
    fl = np.sqrt( fv[0]**2 + fv[1]**2 ) # final center length
    scale = fl/il
    
    scaleMat = np.eye(3)
    scaleMat[0,0] = scale # x-scale
    scaleMat[1,1] = scale # y-scale
    
    # Create Translation Matrix to move to centers
    transMat = np.eye(3)
    transMat[0,2] = g1_cf[0]
    transMat[1,2] = g1_cf[1] 
    
    # Create single adjustment matrix for effeciency        
    adjMat = np.copy( rotMat )    # rotate particles
    adjMat = np.dot( scaleMat, adjMat )  # Scale up values
    adjMat = np.dot( transMat, adjMat )  # translate to new centers
    
        
    # APPLYING MATRIX TO POINTS
    n = g1pts.shape[0]

    # Shift center points  
    sgc = np.transpose( np.dot(adjMat, gc) )

    # One's are required in 3rd dimension for proper translation
    tPts = np.ones((3,n))  
    sg1pts = np.zeros((n,3))
    sg2pts = np.zeros((n,3))

    # Shift galaxy 1
    tPts[0:2,:] = g1pts[:,0:2].T
    sg1pts[:,0:2] = np.matmul( adjMat, tPts )[0:2,:].T  
    sg1pts[:,2] = g1pts[:,2]

    # Shift galaxy 2
    tPts[0:2,:] = g2pts[:,0:2].T  
    sg2pts[:,0:2] = np.matmul( adjMat, tPts )[0:2,:].T    
    sg2pts[:,2] = g2pts[:,2]    
    
    # Sanity Plots and printing
    # ... I've spent waaaaay too many hours trying to write and troubleshoot this 
    devPrint = False
    if devPrint:     
        
        print("\n"+"****"*20)
        print("Image Parameter: %s"%imgArg['name'])
        
        print("Starting Coordinates")
        print("Gal 1: ", g1_ci)
        print("Gal 2: ", g2_ci)
        
        print("Fianl Image Coordinates")
        print("Gal 1: ", g1_mf )
        print("Gal 2: ", g2_mf )
        
        print("Fianl Cartesian Coordinates")
        print("Gal 1: ",g1_cf )
        print("Gal 2: ",g2_cf )
        
        fig, ax = plt.subplots(5,3,figsize=(15,5*6))

        n = g1pts.shape[0]
        print("N points: ",n)
        
        tmp = np.ones((3,2*n))
        tmp[0:2,0:n] = np.copy(g1pts[:,0:2]).T
        tmp[0:2,n:2*n] = np.copy(g2pts[:,0:2]).T

        b = [ -15, 15, -15, 15]
        xedges = np.linspace(b[0],b[1],101)
        yedges = np.linspace(b[2],b[3],201)
        tmp_img1, xedges, yedges = np.histogram2d( tmp[0,:], tmp[1,:], bins=(xedges, yedges), )
        print("Img shape",tmp_img1.shape)
        #tmp_img2, xedges, yedges = np.histogram2d( tmp[0,:], ImgCartConv(tmp[1,:],g2_ci[1]), bins=(xedges, yedges), )
        tmp_img2 = np.rot90( tmp_img1 )

        ax[0,0].set_title("Starting Scatter")
        ax[0,0].scatter(tmp[0,:],tmp[1,:],s=0.5)
        
        ax[0,1].set_title("Starting IMshow")
        ax[0,1].imshow( tmp_img1, cmap='gray', extent=b )
        
        ax[0,2].set_title("Starting IMshow +90")
        ax[0,2].imshow( tmp_img2, cmap='gray', extent=b )
        
        print("Rotation")
        print("Initial Angle: ",np.rad2deg(iv_angle))
        print("Final Angle: ",np.rad2deg(fv_angle))
        
        i = 1
        tmp = np.matmul(rotMat,tmp)

        b = [ -15, 15, -15, 15]
        xedges = np.linspace(b[0],b[1],101)
        yedges = np.linspace(b[2],b[3],101)
        tmp_img1, xedges, yedges = np.histogram2d( tmp[0,:], tmp[1,:], bins=(xedges, yedges), )
        tmp_img2 = np.rot90( tmp_img1 )

        ax[i,0].set_title("Rot 1 Scatter")
        ax[i,0].scatter(tmp[0,:],tmp[1,:],s=0.5)
        
        ax[i,1].set_title("Rot 1 IMshow")
        ax[i,1].imshow( tmp_img1, cmap='gray', extent=b )
        
        ax[i,2].set_title("Rot 1 IMshow+90")
        ax[i,2].imshow( tmp_img2, cmap='gray', extent=b )
        
        print("Rotation")
        print("Initial Angle: ",np.rad2deg(iv_angle))
        print("Final Angle: ",np.rad2deg(fv_angle))
        
        i = 2
        tmp = np.matmul(scaleMat,tmp)

        b = [ -500, 500, -500, 500]
        xedges = np.linspace(b[0],b[1],101)
        yedges = np.linspace(b[2],b[3],101)
        tmp_img1, xedges, yedges = np.histogram2d( tmp[0,:], tmp[1,:], bins=(xedges, yedges), )
        tmp_img2 = np.rot90( tmp_img1 )

        ax[i,0].set_title("Scaling Scatter")
        ax[i,0].scatter(tmp[0,:],tmp[1,:],s=0.5)
        
        ax[i,1].set_title("Scaling IMshow")
        ax[i,1].imshow( tmp_img1, cmap='gray', extent=b )
        
        ax[i,2].set_title("Scaling IMshow+90")
        ax[i,2].imshow( tmp_img2, cmap='gray', extent=b )
        
        i = 3
        tmp = np.matmul(transMat,tmp)

        b = [ 0, 1000, 0, 1000]
        xedges = np.linspace(b[0],b[1],101)
        yedges = np.linspace(b[2],b[3],101)
        tmp_img1, xedges, yedges = np.histogram2d( tmp[0,:], tmp[1,:], bins=(xedges, yedges), )
        tmp_img2 = np.rot90( tmp_img1 )

        ax[i,0].set_title("Translation Scatter")
        ax[i,0].scatter(tmp[0,:],tmp[1,:],s=0.5)
        
        ax[i,1].set_title("Translation IMshow")
        ax[i,1].imshow( tmp_img1, cmap='gray', extent=b )
        
        ax[i,2].set_title("Translation IMshow+90")
        ax[i,2].imshow( tmp_img2, cmap='gray', extent=b )
        
        i = 4
        tmp = np.matmul(adjMat,tmp)

        b = [ 0, 1000, 0, 1000]
        xedges = np.linspace(b[0],b[1],101)
        yedges = np.linspace(b[2],b[3],101)
        tmp_img1, xedges, yedges = np.histogram2d( sg1pts[:,0], sg1pts[:,1], bins=(xedges, yedges), )
        tmp_img2, xedges, yedges = np.histogram2d( sg2pts[:,0], sg2pts[:,1], bins=(xedges, yedges), )
        tmp_img3 = tmp_img1 + tmp_img2
        tmp_img4 = np.rot90( tmp_img3 )

        ax[i,0].set_title("Final Scatter")
        ax[i,0].scatter(sg1pts[:,0],sg1pts[:,1],s=0.5)
        ax[i,0].scatter(sg2pts[:,0],sg2pts[:,1],s=0.5)
        
        ax[i,1].set_title("Final IMshow")
        ax[i,1].imshow( tmp_img3, cmap='gray', extent=b )
        
        ax[i,2].set_title("Final IMshow+90")
        ax[i,2].imshow( tmp_img4, cmap='gray', extent=b )
    
    return sg1pts, sg2pts, gc
# END SHIFTING POINTSSSSSSSS.

def addGalaxy( ptSet, imgArg, gNum=None ):

    width = int(imgArg['image_size']['width'])
    height = int(imgArg['image_size']['height'])
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
    xedges = np.linspace(0,width,width+1)
    yedges = np.linspace(0,height,height+1)
    img, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), weights = weights)
    
    # For some reason, image needs a rotation after histogram.
    img = np.rot90(img)
    
    return img
# End adding galaxy to a image

# add Center circles
def addCircles(img, imgParam, cSize = 7):
    
    im.pprint(imgParam)
    g1c = ( int(imgParam['galaxy_centers']['px']) , int(imgParam['galaxy_centers']['py']) ) 
    g2c = ( int(imgParam['galaxy_centers']['sx']) , int(imgParam['galaxy_centers']['sy']) )
    
    cimg = np.copy(img)
   
    cv2.circle( cimg, g1c, cSize, (255, 255, 255), 2 ) 
    cv2.circle( cimg, g2c, cSize, (255, 255, 255), 2 )
    
    return cimg

# Function for modifying base target image
def adjustTargetImage( tInfo, new_param, startingImg = 'zoo_0', printAll = False ):
    
    tImg = tInfo.getTargetImage( startingImg )
    tParams = tInfo.getImageParams()
    
    old_param = tParams[startingImg]
    if printAll:
        print("\nIC: Adusting Starting Target Image\n")
        #gm.pprint(sParam)
    
    # Define function
    def Centers2Points( param ):
        
        # Initialize points
        pts = np.float32( np.zeros((3,2)) )
    
        # First two points are galaxy centers
        px = int( param['imgArg']['galaxy_centers']['px'] )
        py = int( param['imgArg']['galaxy_centers']['py'] )
        sx = int( param['imgArg']['galaxy_centers']['sx'] )
        sy = int( param['imgArg']['galaxy_centers']['sy'] )

        pts[0,0] = px 
        pts[0,1] = py 
        pts[1,0] = sx 
        pts[1,1] = sy 

        # Create right triangle with 3rd point
        pts[2,0] =  px - (py-sy) 
        pts[2,1] =  py + (px-sx) 
        
        return pts
    # End defining fuction
    
    sPts = Centers2Points( old_param )
    fPts = Centers2Points( new_param )

    # Create affine transform matrix out of sets of points
    M = cv2.getAffineTransform(sPts,fPts)
    
    w = int( new_param['imgArg']['image_size']['width'] )
    h = int( new_param['imgArg']['image_size']['height'] )
    
    # Create new target image
    newImg = cv2.warpAffine( tImg, M, ( w, h ) )
    
    # Save new target image
    newLoc = tInfo.findTargetImage( tName = new_param['imgArg']['name'], newImg=True)
    print(newLoc)
    
    cv2.imwrite( newLoc, newImg )


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
