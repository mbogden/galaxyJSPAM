'''
    Author:     Matthew Ogden and Richa
    Created:    19 Oct 2019
    Altered:    13 Nov 2019
Description:    This will train a nueral network to classify images
'''

from os import path, listdir
from sys import exit, argv

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing import image

from mpl_toolkits.mplot3d import axes3d
#get_ipython().magic(u'matplotlib inline')
#from IPython.display import display


#Global Variables

filterDir = ''
printAll = True

def filter_trainer():

    print(filterDir)

    goodDir = filterDir + 'goodImgs/'
    gArTrain, gArTest = getImgs( goodDir )

    badDir = filterDir + 'badImgs/'
    bArTrain, bArTest = getImgs( badDir )

    trainImgs, trainLabels = orgImgsLabels( gArTrain, bArTrain )
    testImgs, testLabels = orgImgsLabels( gArTest, bArTest )

    nTrain = len( trainImgs )
    nTest = len( testImgs )

    print("number of train: %d - test: %d" % ( nTrain, nTest ) )

    model = trainModel( trainImgs, trainLabels )

    # Evaluate score 

    score = model.evaluate(testImgs, testLabels, verbose=1)
    print('Test accuracy:', score[1])


    # Save model to JSON
    model_json = model.to_json()
    with open( filterDir + "filter_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filterDir + "filter_model.h5")
    print("Saved model to disk")
     
# End main

def trainModel( trainImgs, trainLabels ):
 
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(3,3),
                                  activation='relu',
                                  input_shape=[trainImgs.shape[1],
                                               trainImgs.shape[2],
                                               trainImgs.shape[3]]))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128,128,1)))
    model.add(keras.layers.ELU())
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128,128,1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())




    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr = 0.0001),
                  metrics=['accuracy'])
    model.summary()

    batch_size = 5
    epochs = 10
    history = model.fit(trainImgs, trainLabels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split = 0.2)

    return model


def old():
    pass

    # later...
     
    # load json and create model
    #json_file = open('model3.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("model3.h5")
    #print("Loaded model from disk")

    # evaluate loaded model on test data
    #loaded_model.compile(loss=keras.losses.categorical_crossentropy,
    #              optimizer=keras.optimizers.Adam(),
    #              metrics=['accuracy'])
    #score = loaded_model.evaluate(data_test, data_test_labels, verbose=1)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])


    # In[39]:


    #a = os.listdir('./training_image_set_2/goodImgs')
    #for f in a:
    #    i = grab_image('./training_image_set_2/goodImgs/'+f)
    #    print('File name:', f, ' Prediction:', model.predict(i))
        
    #print('/n/n/n/n')

    #b = os.listdir('./training_image_set_2/badImgs')
    #for f in b:
    #    i = grab_image('./training_image_set_2/badImgs/'+f)
    #    print('File name:', f, ' Prediction:', model.predict(i))


def orgImgsLabels( gImgs, bImgs ):

    labelList = []

    for g in gImgs:
        labelList.append( 0 )

    for b in bImgs:
        labelList.append( 1 ) 
    
    labels = keras.utils.to_categorical(labelList)

    imgs = np.concatenate(( gImgs, bImgs))

    if len( imgs ) != len( labelList ):
        print("BAD!")
        exit()

    return imgs, labels



def getImgs( imgDir ):

    if not path.exists( imgDir ):
        print("Couldn't find badDir")
        print("Exiting...")
        exit()

    imgNames = listdir(imgDir)

    trainingImgs = []
    testingImgs = []

    for i,f in enumerate(imgNames):
        imgLoc = imgDir + f

        if i % 2 == 0:
            trainingImgs.append( grab_image( imgLoc ) )
        else:
            testingImgs.append( grab_image( imgLoc ) )

    trainingAr = np.concatenate(trainingImgs)
    testingAr = np.concatenate(testingImgs)

    return trainingAr, testingAr

# end getting images for training filter format

def grab_image(img_path):

    img = image.load_img(img_path, target_size=(128, 128), color_mode = "grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x
#end grab_image




def readArg( argList ):

    global printAll, filterDir

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-filterDir':
            filterDir = argList[i+1]
            if filterDir[-1] != '/': filterDir += '/'

    # Check if input arguments are valid
    endEarly = False

    if filterDir == '' or not path.exists( filterDir ):
        print("Could not find filterDir: %s" % filterDir)
        endEarly = True

    return endEarly

# End reading command line arguments

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

# Run main after declaring functions
if __name__ == '__main__':
    argList = argv
    endEarly = readArg(argList)
    if endEarly:
        print("Exiting...")
        exit()
    filter_trainer()
