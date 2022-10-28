#!/usr/bin/env python2

import wndcharm
from skimage.io import imread
from skimage.transform import resize
from wndcharm.FeatureVector import FeatureVector
from wndcharm.PyImageMatrix import PyImageMatrix

from os import path
import pandas as pd
import numpy as np


#print wndcharm.diagnostics

def test():
    statement = "Hi!  You're in Matthew's code for all things image feature extraction"
    print statement
    return statement

def readList( locFile ):
    
    if not path.exists(locFile):
        return None
        
    imgFile = open(locFile,'r')
    imgLocs = []
    for l in imgFile:
        l = l.strip()
        if path.exists(l):
            imgLocs.append(l)
            
    imgFile.close()
    return imgLocs


def createFeats( img ):
    
    matrix = PyImageMatrix()
    imgShape = img.shape
       
    matrix.allocate(img.shape[1], img.shape[0])

    np_mat = matrix.as_ndarray()
    np_mat[:] = img
    fv = FeatureVector( name='FromNumpyMatrix', long=True, original_px_plane=matrix )
    fv.GenerateFeatures( quiet=True, write_to_disk=False )
    return fv.values
    

def main(locFile, featDir):
    
    print 'locFile: %s' % locFile
    print 'featureDir: %s' % featDir
    
    imgLocList = readList( locFile )
    if imgLocList == None:
        print("No images found")
        return
    
    featAr = np.zeros( (len(imgLocList), 2919) )
    featLoc = featDir + "wndcharm_feats.csv"
    if path.exists(featLoc):
        featAr = pd.read_csv(featLoc).values
        nImg = featAr.shape[0]

    imgInfoList = []
        
    for i, imgLoc in enumerate(imgLocList):
        
        #print('%s : %s' % (i,imgLoc))
        if i >= nImg:
            break

        # Check if feats already generated, if so skip
        valSum = np.sum( featAr[i,:] )
        if valSum > 0.001:
            continue

        try:
            img = imread( imgLoc )
        except:
            print('IMG FAILED:')
            print('\t - runDir: %s' % rInfo.runDir )
            print('\t - imgLoc: %s' % imgLoc )
            continue
        
        sameImgFound = False
        
        for fI, pImg in imgInfoList:
            
            diff = np.amax( np.abs( img - pImg ) )
            
            if diff < 0.001:
                sameImgFound = True
                featAr[i,:] = featAr[fI,:]
                break
        
        if not sameImgFound:            
        
            featAr[i,:] = createFeats( img )
            imgInfoList.append( ( i, img ) )
            
        print('%d / %d' % ( i, len(imgLocList ) ) )

        np.savetxt( featLoc, featAr, delimiter=",")
    
    np.savetxt(featDir + "wndcharm_feats.csv", featAr, delimiter=",")
        
print("Boop")
if __name__ == '__main__':
    
    print("HI")
    
    from sys import argv, exit
    
    '''
    if len(argv) != 3:
        print "not the right arguments"
        exit()
    '''
    
    locFile = argv[1]
    featDir = argv[2]
    
    if not path.exists(locFile):
        print "img location file does not exist"
        exit()
    
    if not path.exists(featDir):
        print "featLoc does not exist"
        exit()    
       
    main(locFile,featDir)
