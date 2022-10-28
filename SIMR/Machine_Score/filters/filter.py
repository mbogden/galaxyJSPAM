'''
    Author:     Matthew Ogden and Richa
    Created:    29 Oct 2019
    Altered:    15 Nov 2019
Description:    For quick implementation of testing nueral network model filter
Ex Execution:   python3 baseFilter.py -imgLoc path/to/img.png -modelJsonLoc model2.json -modelH5Loc model2.h5
'''

from sys import exit, argv
from os import  path, listdir

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.preprocessing import image
from keras.models import model_from_json


# Global Variables
printAll = True
imageSize = ( 128, 128 )

imgLoc = ''
modelJsonLoc = ''
modelH5Loc = ''
modelDir = ''

'''
imageLoc = './587727222471131318_bad_test/587727222471131318_223_init_diff.png'
modelJsonLoc = './model2.json'
modelH5Loc = './model2.h5'
'''



def main():

    if printAll:
        print("Image Loc: %s" % imgLoc)
        print("Model Json Loc: %s" % modelJsonLoc)
        print("Model H5 Loc: %s" % modelH5Loc)

    # load the Keras model
    loadedModel = loadModelFunc(modelJsonLoc, modelH5Loc)

    # Load the image
    image = loadImage( imgLoc )

    # Get prediction
    prediction = predictImg( image, loadedModel )

    if printAll: print( prediction )

# End main


def convertCV2Img( imgIn ):

    import cv2

    if type(imgIn) is not list:
 
        img = cv2.resize( img, imageSize )
        img = image.img_to_array( img )
        img = np.expand_dims( img, axis=0)
        return img
    
    else:
    
        newImgs = []

        for img in imgIn:

            img = cv2.resize( img, imageSize )
            img = image.img_to_array( img )
            img = np.expand_dims( img, axis=0)

            newImgs.append( img )

        return newImgs

        

def loadFilterDir( fDir ):
    
    fList = listdir( fDir )

    jsonLoc = ''
    h5Loc = ''

    for f in fList:
        if '.json' in f:
            jsonLoc = fDir + f

        elif '.h5' in f:
            h5Loc = fDir + f
    # end going through dir

    return loadModelFunc( jsonLoc, h5Loc )


def loadModelFunc( modelJsonLoc, modelH5Loc ):
    
    
    # load json and create model
    jsonIn = open(modelJsonLoc, 'r')

    loaded_model = model_from_json( jsonIn.read() )


    # load weights into new model
    loaded_model.load_weights(modelH5Loc)
    if printAll: print("Loading Keras model")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    if printAll: print("Loaded model from disk")

    return loaded_model
# End loading model


def loadImage( imageLoc ):
    
    if printAll: print("Loading image")
    
    img = image.load_img(imageLoc, target_size=imageSize, color_mode = "grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def predictImg( imgs, loadedModel ):

    if type(imgs) is not list:
        return loadedModel.predict_classes( imgs )[0] 
    
    else:
        pList = []
        for img in imgs:
            pList.append( loadedModel.predict_classes( img )[0] )
        return pList

def readArg(argList):

    global printAll, imgLoc, modelJsonLoc, modelH5Loc

    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-imgLoc':
            imgLoc = argList[i+1]

        elif arg == '-jsonLoc':
            modelJsonLoc = argList[i+1]

        elif arg == '-h5Loc':
            modelH5Loc = argList[i+1]

    # Check if input arguments are valid
    
    if imgLoc == '' or not path.exists( imgLoc ):
        print( "Image not found: %s" % imgLoc )
        endEarly = True

    if modelJsonLoc == '' or not path.exists( modelJsonLoc ):
        print( "Image not found: %s" % modelJsonLoc )
        endEarly = True

    if modelH5Loc == '' or not path.exists( modelH5Loc ):
        print( "Image not found: %s" % modelH5Loc )
        endEarly = True

    return endEarly

# End reading command line arguments

if __name__=='__main__':

    endEarly = readArg(argv)
    if endEarly:
        print("Exiting....")
        exit()

    main()

