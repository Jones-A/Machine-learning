# This is the release version free to send to others
#This is a small CNN meant to recognize spikes in data images we generate
import numpy as np
#import dask as dk removed from this version for sending to others
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,LeakyRelu,ReLu,BatchNormalization
import cv2
import json

path = "/isilon/datalake/neuroanatomy/original/Machine_Learning"
validation_path = path + "/validation_set"
os.chdir(path)
trainingSpike=[]
trainingFlat=[]
validationSpike=[]
validationFlat=[]
#data
#data set is images of the sorted spikes both 9x9 and single individual spikes
# I used opencv in another location to edit the original data set of about two 750 image per class(spike and non spike)so there are about 925 images per class
#This should be a function
os.chdir("./train_set/spikes")
files= os.listdir(".")
for img in files
    curImg= cv2.imread(img.path, cv2.IMREAD_GRAYSCALE)
    trainingSpike.append(cv2.resize(curImg,(224,224),cv2.IMREAD_GRAYSCALE))

os.chdir("../flat")
files= os.listdir(".")
for img1 in files
    curImg1= cv2.imread(img1.path, cv2.IMREAD_GRAYSCALE)
    trainingFlat.append(cv2.resize(curImg1,(224,224),cv2.IMREAD_GRAYSCALE))

os.chdir(validation_path)
os.chdir("../spikes")
files= os.listdir(".")
for img2 in files
    curImg2= cv2.imread(img2.path, cv2.IMREAD_GRAYSCALE)
    validationSpike.append(cv2.resize(curImg2,(224,224),cv2.IMREAD_GRAYSCALE))

os.chdir("../flat")
files= os.listdir(".")
for img3 in files
    curImg3 = cv2.imread(img.path, cv2.IMREAD_GRAYSCALE)
    validationFlat.append(cv2.resize(curImg3,(224,224),cv2.IMREAD_GRAYSCALE))


np.asarray(validationFlat)
np.asarray(validationSpike)
np.asarray(trainingFlat)
np.asarray(trainingSpike)

fullTrain = np.concatenate([trainingSpike,trainingFlat])
fullValid = np.concatenate([validationSpike,validationFlat])

#normalization our files are alreadyhomgenous stnadardization is not worth the extra computation time for standardization
fullTrain = fullTrain/255
fullValid = fullValid/255

#label stamping
# trainLabel creates a vector of the size of the input vector settig the first few entries to the given value and the rest to zero or 1 respectively
trainLabel=simpleStamper(fullTrain,940,1)
validLabel=simpleStamper(fullValid,80,0)

os.chdir(path)
#final cnn and deep network step
#def
model = Sequential()
#convolution layer
model.add(Conv2D(8,kernel_size=(3,3),strides=(1,1),padding="same",input_shape=(224,224,1),activation='relu')) # we have so little data that a small stride is computationally feasible
model.add(MaxPooling2D(pool_size(2,2),strides=(3,3),padding="same"))
model.add(Conv2D(16,kernel_size=(1,1),strides=(1,1),padding="same",activation="leakyrelu"))
model.add(MaxPooling2D(pool_size(2,2),strides=(1,1),padding="same"))
model.add(Flatten())
#shallow deep network...oxymoron
model.add(Dense(512),activation='leakyrelu')
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(257))
#output
model.add(Dense(1,activation="softmax"))
#compile
module.compile(loss='binary_crossentropy',optimizer='Adagrad',metrics=['accuracy'])

#json because I wanna try putting this on aws one day
json_string = model.to_json()
with open('model.json','w') as modelDef
    modelDef.write(json_string)
#run it
model.fit(fullTrain,trainLabel,epochs=16,verbose=2,batch_size=16,validation_data=(fullValid,validLabel),shuffle=True))


