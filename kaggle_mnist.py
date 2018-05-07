# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:09:58 2018

@author: uhyy
"""

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split

from keras.utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Nadam
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
#load data
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
Y_train = train["label"]
Y_train.value_counts()
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 
# Check the data
X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

#-------------------------------------------
#---------MODEL-----------------------------
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 64
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = Nadam() , loss = "categorical_crossentropy", metrics=["accuracy"])

"""
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
"""
model_checkpoint = ModelCheckpoint('cnn.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(X_train,Y_train, batch_size=batch_size,
                              epochs = epochs, validation_split=0.2, shuffle=True,
                              verbose = 1, callbacks=[model_checkpoint])
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist.csv",index=False)
