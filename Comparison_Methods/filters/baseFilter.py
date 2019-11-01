'''
    Author:     Matthew Ogden
    Created:    29 Oct 2019
    Altered:    29 Oct 2019
Description:    For quick implementation of testing nueral network model filter
Ex Execution:   python3 baseFilter.py -imageLoc path/to/img.png -modelLoc model.h5
'''

from sys import \
        exit, \
        argv

from os import \
        path, \
        listdir

printAll = True
imageLoc = ''
modelLoc = ''

def main(argList):

    endEarly = readArg(argList)

    if printAll:
        print("Image Loc: %s" % imageLoc)
        print("Model Loc: %s" % modelLoc)

    if endEarly:
        print("Exiting...")
        exit(-1)

    # load the Keras model
    loadedModel = loadModelFunc(modelLoc)

    # Load the image
    image = loadImage( imageLoc )

    # Get prediction
    prediction = predictImg( image, loadedModal )

    # do something with prediction
    doSomething( prediction )

# End main

def loadModelFunc( modelLoc ):

    print("Loading Keras model")

    return 'keras model class thing'



def loadImage( imageLoc ):

    print("Loading image")

    return "image in needed format for model prediction"


def predictImg( image, loadedModel ):
    print("Making prediction")

    # Take image and get prediction from keras model

    return 'keras prediction'

def doSomething( prediction ):

    # Just print the prediction contents for now to check if working
    print(prediction)

def readArg(argList):

    global printAll, imageLoc, modelLoc

    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-imageLoc':
            imageLoc = arglist[i+1]

        elif arg == '-modelLoc':
            modelLoc = arglist[i+1]

    # Check if input arguments are valid

    if not path.exists(imageLoc):
        print("Image does not exist: %s" % imageLoc)
        endEarly = True

    if not path.exists(modelLoc):
        print("Keras model does not exist: %s" % modelLoc)
        endEarly = True

    return endEarly

# End reading command line arguments


def readArgFile(argList, argFileLoc):

    try:
        argFile = open( argFileLoc, 'r')
    except:
        print("Failed to open/read argument file '%s'" % argFileLoc)
    else:

        for l in argFile:
            l = l.strip()

            # Skip line if empty
            if len(l) == 0:
                continue

            # Skip line if comment
            if l[0] == '#':
                continue

            lineItems = l.split()
            for item in lineItems:
                argList.append(item)

        # End going through file

    return argList
# end read argument file


def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return False, []
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return False, []

    else:
        inList = list(inFile)
        inFile.close()
        return True, inList

# End simple read file

# Run main after declaring functions
if __name__ == '__main__':
    argList = argv
    main(argList)
