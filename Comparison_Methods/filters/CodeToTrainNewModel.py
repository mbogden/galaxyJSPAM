#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras import backend as K
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
get_ipython().magic(u'matplotlib inline')
from IPython.display import display

import os, sys, csv


# In[2]:


gooddir = './training_image_set_2/goodImgs'
baddir = './training_image_set_2/badImgs'
goodImgs = os.listdir(gooddir)
badImgs = os.listdir(baddir)


# In[3]:


goodImgs.sort()
badImgs.sort()


# In[10]:


i = 0
trainingGood = []
testingGood = []
trainingBad = []
testingBad = []
for f in goodImgs:
    if i % 2 == 0:
        trainingGood.append(f)
    elif i % 2 != 0:
        testingGood.append(f)
    i += 1


# In[13]:


j = 0
trainingBad = []
testingBad = []
for f in badImgs:
    if j % 2 == 0:
        trainingBad.append(f)
    elif j % 2 != 0:
        testingBad.append(f)
    j += 1


# In[19]:


def grab_image(img_path):
    #path = dirs+'/'+img_path
    img = image.load_img(img_path, target_size=(128, 128), color_mode = "grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


# In[22]:


trainingGoodArr = np.concatenate([grab_image('./training_image_set_2/goodImgs/'+f) for f in trainingGood])
testingGoodArr = np.concatenate([grab_image('./training_image_set_2/goodImgs/'+f) for f in testingGood])
trainingBadArr = np.concatenate([grab_image('./training_image_set_2/badImgs/'+f) for f in trainingBad])
testingBadArr = np.concatenate([grab_image('./training_image_set_2/badImgs/'+f) for f in testingBad])


# In[24]:


training = np.concatenate((trainingGoodArr, trainingBadArr))
testing = np.concatenate((testingGoodArr, testingBadArr))


# In[25]:


import csv
goodCSVData = ['0']
badCSVData = ['1']
with open('labels2.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for i in trainingGood: #goodtrain
        writer.writerows(goodCSVData)
    for i in trainingBad: #badtrain
        writer.writerows(badCSVData)
    for i in testingGood:
        writer.writerows(goodCSVData)
    for i in testingBad:
        writer.writerows(badCSVData)
csvFile.close()


# In[26]:


labels = []
with open('labels2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\n')
    for row in readCSV:
        labels.append(row[0])


# In[27]:


Y = keras.utils.to_categorical(labels)
Y.shape


# In[28]:


n = 0
m = 0
for f in trainingGood:
    n += 1
for f in trainingBad:
    n += 1
for f in testingGood:
    m += 1
for f in testingBad:
    m += 1
print(n)
print(m)


# In[29]:


train_labels = Y[ : n]
test_labels = Y[n :]
print(training.shape)
print(testing.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[30]:


model = keras.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(3,3),
                              activation='relu',
                              input_shape=[training.shape[1],
                                           training.shape[2],
                                           training.shape[3]]))
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


# In[35]:


batch_size = 5
epochs = 10
history = model.fit(training, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split = 0.2)


# In[36]:


score = model.evaluate(testing, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[38]:


#from keras.models import model_from_json

# serialize model to JSON
#model_json = model.to_json()
#with open("model3.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model3.h5")
#print("Saved model to disk")
 
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


# In[ ]:




