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
from mpi4py import MPI

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


def main_ic_run( rInfo = None, arg = gm.inArgClass() ):
    
    # extract variables
    rDir = arg.runDir
    printBase=arg.printBase
    printAll=arg.printAll
    overWrite = arg.get('overWrite',False)
    
    if rInfo == None:
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
        if printBase: print('IC: WARNGING:\n\t - rInfo status not good. Exiting...' )
        return
    
    if scoreParams == None:
        if printBase: print("IC: WARNING: Please provide score parameters")
        return
    
    elif printAll:
        print("IC: given parameters: %d"%len(scoreParams))
    
    # Extract types of images being created
    imgParams = {}
    
    for pKey in scoreParams:
        imgKey = scoreParams[pKey]['imgArg']['name']
        if imgKey not in imgParams:
            imgParams[imgKey] = scoreParams[pKey]
            
    # Loop through files
    n = len(scoreParams)
    for i, pKey in enumerate(imgParams):
        if printAll: print( "IC: ",pKey )
                    
        sParam = imgParams[pKey]
        create_image_from_parameters( rInfo, sParam, printAll = printAll, overwrite=overWrite, )
        
        if printBase:
            print( 'IC_LOOP: %4d / %4d' % (i+1,n), end='\r' )
    if printBase: print( 'IC_LOOP: %4d / %4d: COMPLETE' % (n,n) )
            
# End main image creator run
        
def create_image_from_parameters( rInfo, sParam, overwrite=False, printAll = False, ):
    
    if printAll: 
        print("IC: Creating image:")
        im.tabprint('runId: %s'%rInfo.get('run_id'))
        im.tabprint('score: %s'%sParam.get('name'))
        
    # Add place to keep points and images in memory
    if rInfo.get('img',None) == None:
        rInfo.img = {}
        
    # Add place to keep points and images in memory
    if rInfo.get('init',None) == None:
        rInfo.init = {}
        
    if rInfo.get('pts',None) == None:
        rInfo.pts = {}
    
    # Grab idenitfying names
    imgName = sParam['imgArg'].get('name',None)
    imgType = sParam['imgArg'].get('type','model')
    simName = sParam['simArg'].get('name',None)

    if imgType == 'model':
        # If image is already created, return
        mImgLoc = rInfo.findImgLoc( imgName, imgType='model')
        iImgLoc = rInfo.findImgLoc( imgName, imgType='init' )

        # Check if images already created
        if mImgLoc != None and iImgLoc != None:        
            if printAll: print("IC: Image '%s' already made for %s"%(imgName,rInfo.get('run_id')))

            # Unless overwriting images, load images and return
            if not overwrite: 
                mImg = gm.readImg( mImgLoc )
                iImg = gm.readImg( iImgLoc )
                return
            
    elif imgType == 'wndchrm':
        wImgLoc = rInfo.findImgLoc( imgName, imgType='wndchrm')
        mImgLoc = rInfo.findImgLoc( imgName, )
    
        # Check if images already created
        if wImgLoc != None and mImgLoc != None and not overwrite:        
            if printAll: print("IC: Image '%s' already made for %s"%(imgName,rInfo.get('run_id')))
            return
            
    # Get particles
    pts = getParticles( rInfo, simName, printAll=printAll )    
    if type( pts ) == type( None ):
        if rInfo.printBase: print("WARNING: IC: Exiting Image Creation")
        return None
    
    if printAll: im.tabprint("Creating image from points")

    imgArg = sParam['imgArg']
    
    # Add particles to image
    mImg = pts2image( pts, imgArg, init=False )
    iImg = pts2image( pts, imgArg, init=True )

    # Apply blur
    mImg = blurImg( mImg, imgArg )
    iImg = blurImg( iImg, imgArg )

    # Normalize brightness
    mImg = normImg( mImg, imgArg )
    iImg = normImg( iImg, imgArg )
    

    # Use image as float32 type for scoring
    if mImg.dtype == np.uint8:
        mImg = gm.uint8_to_float32( mImg )
        rInfo.img[imgName] = mImg

    if mImg.dtype == np.uint8:
        iImg = gm.uint8_to_float32( iImg )
        rInfo.init[imgName] = iImg

    # Get Image locations
    mImgLoc = rInfo.findImgLoc( imgName, newImg = True )
    iImgLoc = rInfo.findImgLoc( imgName, newImg = True, imgType = 'init' )

    if printAll: 
        im.tabprint("Saving model image at: %s"%mImgLoc)
        im.tabprint("Saving unperturbed at: %s"%iImgLoc)
            
    # Save image
    gm.saveImg(mImgLoc,mImg)
    gm.saveImg(iImgLoc,iImg)
        
    # If type wndchrm, also place in wndchrm folder as tiff. 
    if imgType ==  'wndchrm': 
        if np.amax( mImg.shape ) > 100:
            if rInfo.printBase: im.tabprint("WARNING: WNDCHRM images over 100 pixels not allowed.")
            return
        
        wImgLoc = rInfo.findImgLoc( imgName, newImg = True, imgType = 'wndchrm' )
        if printAll:    im.tabprint("Saving wndchrm image : %s"%wImgLoc)
        gm.saveImg( wImgLoc, mImg )

# End image from score parameter file


def getParticles( rInfo, simName, printAll=False ):
    
    # Check if files are loaded in run info class
    pts = rInfo.pts.get(simName,None)
    
    # Load points if not found
    if pts != None:
        return pts
    
    # Get particle location 
    ptsZipLoc = rInfo.findPtsLoc( ptsName = simName )
    if printAll: im.tabprint("Loading points from file: %s"%ptsZipLoc)
        
    if ptsZipLoc == None:
        if rInfo.printBase: print("WARNING: IC: zipped points not found: %s"%ptsZipLoc)
        return None
    
    # Read particles using rInfo
    pts_i, pts_f = rInfo.getParticles( simName )
    
    if type(pts_i) == type(None) or type( pts_f ) == type(None): 
        if rInfo.printBase: print("WARNING: IC: zipped points failed to read: %s"%ptsZipLoc)
        return None
    
    # Analyze points with particle class
    pts = particle_class( pts_i, pts_f )  
    
    # Save pts in case needed for later
    rInfo.pts[simName] = pts
    
    return pts

# End getting particles
    
    
def pts2image( pts, imgArg, init=False ):
    
    # Final model image
    if not init:
        pts.sg1f, pts.sg2f, pts.sfCenters = shiftPoints( imgArg, pts.g1f, pts.g2f, pts.fCenters, pts.fCenters )
        g1img = addGalaxy( pts.sg1f, imgArg, 0 )
        g2img = addGalaxy( pts.sg2f, imgArg, 1 )
        img = g1img + g2img
        
    else:
        # TO Initial image
        pts.sg1i, pts.sg2i, pts.sfCenters = shiftPoints( imgArg, pts.g1i, pts.g2i, pts.iCenters, pts.fCenters )
        g1img = addGalaxy( pts.sg1i, imgArg, 0 )
        g2img = addGalaxy( pts.sg2i, imgArg, 1 )         
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
        normArg = {'max_brightness':1.0}
        
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
    
    # exponential curve, dim pixels get much brighter
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
    
    def __init__( self, pts_i, pts_f, printAll = False ):
                
        self.printAll = printAll
        
        # Read from txt files
        self.g1i, self.g2i, self.iCenters = self.extractPts( pts_i )
        self.g1f, self.g2f, self.fCenters = self.extractPts( pts_f )
        
        if printAll:
            print(self.g1i.shape, self.g2i.shape, self.iCenters.shape)
            print(self.g1f.shape, self.g2f.shape, self.fCenters.shape)
            
    # End initializing particle files.

    
    def extractPts( self, pts, ):
        
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


# It finally works... Please never touch this again. 
def shiftPoints( imgArg, g1pts, g2pts, iPtsCenters, fPtsCenters, devPrint=False ):
    
    # Grab number of particles per galaxy
    n = g1pts.shape[0]
    
    # Grab initial values
    imgHeight = imgArg['image_size']['height']
    
    # If shifting initial galaxies to final location, create addition matrix
    g2_shift = fPtsCenters[1,0:2] - iPtsCenters[1,0:2]
    
    if np.sum( np.abs( g2_shift) ) > 1e-6:
        g2_full_shift = np.ones((2,n)) 
        g2_full_shift[0,:] *= g2_shift[0]
        g2_full_shift[1,:] *= g2_shift[1]
    
    # Final image (m) coordinates for galaxy centers
    g1_mf = np.zeros(2)
    g2_mf = np.zeros(2)
    g1_mf[0] = imgArg['galaxy_centers']['px']
    g1_mf[1] = imgArg['galaxy_centers']['py']
    g2_mf[0] = imgArg['galaxy_centers']['sx']
    g2_mf[1] = imgArg['galaxy_centers']['sy']
    
    # Initial Cartesian (c) coordinates for galaxy centers
    g1_ci = fPtsCenters[0,0:2]
    g2_ci = fPtsCenters[1,0:2]
    
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
    
    # If initial unperturbed image, adjust galaxy 2 to final position
    if np.sum( np.abs( g2_shift) ) > 1e-6:
        tPts[0:2,:] += g2_full_shift
    
    sg2pts[:,0:2] = np.matmul( adjMat, tPts )[0:2,:].T    
    sg2pts[:,2] = g2pts[:,2]    
    
    # Sanity Plots and printing
    # I've spent waaaaay too many hours trying to troubleshoot this function...
    # So I finally decided to print and plot every agonizing step.
    # Turns out, it was the histogram2D function used else where that was the issue.
    devPrint = False
    if devPrint:     
        
        print("\n"+"****"*20)
        print("Image Parameter: %s"%imgArg['name'])
        
        print("Initial Coordinates")
        print(iPtsCenters)
        
        print("Final Coordinates")
        print(fPtsCenters)
        
        print("Shifting Galaxy 2")
        print(g2_shift)
        
        print("Starting Coordinates")
        print("Gal 1: ", g1_ci)
        print("Gal 2: ", g2_ci)
        
        print("Fianl Image Coordinates")
        print("Gal 1: ", g1_mf )
        print("Gal 2: ", g2_mf )
        
        print("Fianl Cartesian Coordinates")
        print("Gal 1: ",g1_cf )
        print("Gal 2: ",g2_cf )
        
        fig, ax = plt.subplots(6,3,figsize=(15,5*6))

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
        
        i=0
        ax[i,0].set_title("Starting Scatter")
        ax[i,0].scatter(tmp[0,:],tmp[1,:],s=0.5)
        
        ax[i,1].set_title("Starting IMshow")
        ax[i,1].imshow( tmp_img1, cmap='gray', extent=b )
        
        ax[i,2].set_title("Starting IMshow +90")
        ax[i,2].imshow( tmp_img2, cmap='gray', extent=b )
        
        print("Rotation")
        print("Initial Angle: ",np.rad2deg(iv_angle))
        print("Final Angle: ",np.rad2deg(fv_angle))
        
        
        i += 1
        # Shift particles if unperturbed image
        
        if np.sum( np.abs( g2_shift) ) > 1e-6:
            g2_full_shift = np.ones((2,n)) 
            g2_full_shift[0,:] *= g2_shift[0]
            g2_full_shift[1,:] *= g2_shift[1]
            print("Full shift\n",g2_full_shift)
            tmp[0:2,n:2*n] += g2_full_shift

        b = [ -15, 15, -15, 15]
        xedges = np.linspace(b[0],b[1],101)
        yedges = np.linspace(b[2],b[3],101)
        tmp_img1, xedges, yedges = np.histogram2d( tmp[0,:], tmp[1,:], bins=(xedges, yedges), )
        tmp_img2 = np.rot90( tmp_img1 )

        ax[i,0].set_title("Shift Galaxy 2 Scatter")
        ax[i,0].scatter(tmp[0,:],tmp[1,:],s=0.5)
        
        ax[i,1].set_title("Shift Galaxy 2 IMshow")
        ax[i,1].imshow( tmp_img1, cmap='gray', extent=b )
        
        ax[i,2].set_title("Shift Galaxy 2 IMshow+90")
        ax[i,2].imshow( tmp_img2, cmap='gray', extent=b )
        
        i += 1
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
        
        i += 1
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
        
        i += 1
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
        
        i += 1
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
def adjustTargetImage( tInfo, new_param, startingImg = 'zoo_0', printAll = False, overWrite=False ):
    
    if printAll:
        print("\nIC: Adusting Starting Target Image\n")
        #gm.pprint(sParam)
        
    tImg = tInfo.getTargetImage( startingImg )
    tParams = tInfo.getImageParams()    
    old_param = tParams.get(startingImg,None)
    
    # Check if starting img is valid
    if type(tImg) == type(None) or old_param == None:
        print("WARNING: IC: adjustTargetImage:")
        gm.tabprint("Previous image and/or params invalid")
        gm.tabprint("Image: %s"%type(tImg))
        gm.tabprint("Param: %s"%type(old_param))
    
    # Check if target image is already created
    newName = new_param['imgArg']['name']
    if newName in tParams and not overWrite:        
        if printAll: gm.tabprint("Target image already made: %s"%newName)
            
        return
    
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
    
    if printAll:
        gm.tabprint("From points:")
        print(sPts)
        gm.tabprint("To points:")
        print(fPts)

    # Create affine transform matrix out of sets of points
    M = cv2.getAffineTransform(sPts,fPts)
    
    w = int( new_param['imgArg']['image_size']['width'] )
    h = int( new_param['imgArg']['image_size']['height'] )
    
    if printAll:
        gm.tabprint("Warp Matrix")
        print(M)
    
    # Create new target image
    newImg = cv2.warpAffine( tImg, M, ( w, h ) )
    
    # Create location for new image
    newLoc = tInfo.findTargetImage( tName = new_param['imgArg']['name'], newImg=True)
    
    if printAll:
        gm.tabprint("Writing to loc: %s"%newLoc)
    
    # Write image to location    
    gm.saveImg( newLoc, newImg )
    
    if printAll:
        gm.tabprint("File should exist: %s"%gm.validPath(newLoc))
    
    # Have tInfo load image
    tImg = tInfo.getTargetImage( new_param['imgArg']['name'], overwrite=True )
    
    return tImg


def plot_run_images( rInfo, group_param, nCol = 3 ):
    from math import ceil
    
    i = 0
    j = 0
    n = len( group_param )
    nRow = ceil( n / nCol )
    if nRow < 2:
        nRow = 2
    
    fig, ax = plt.subplots( nRow, nCol ,figsize=(nCol*4,nRow*4) )
    
    for pKey in group_param:
        
        imgName = group_param[pKey]['imgArg']['name']
        
        ax[i,j].set_title(imgName)
        img = rInfo.getModelImage( imgName, overWrite = True, toType = np.uint8)
        if type(img) == type(None):
            print("image warning: None")
        elif img.dtype != np.uint8:
            print("img warning:",img.dtype)
        else:
            ax[i,j].imshow( img, cmap='gray' )
        
        j += 1
        if j >= nCol:
            j = 0
            i += 1    
    


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
