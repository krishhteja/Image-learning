import time
import numpy as np
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard

gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) #Use only 1/3rd of gpu memory
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions))

dataDir = 'PATH_TO_TRAINING_DATA'
categories = []
for category in os.listdir(dataDir):
    categories.append(category) #Generate the categories list using directory names
print(categories)

trainingData = []
imageSize = 50

def createTrainingData():
    print("Creating training data")
    for category in categories:
        path = os.path.join(dataDir, category)
        categoryIndex = categories.index(category)
        for image in os.listdir(path):
            try:
                imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)  # convert image to grayscale
                #plt.imshow(imageArray, cmap='gray')  # Print grayscale image
                resizeArray = cv2.resize(imageArray, (imageSize, imageSize))  # Converting all images to 50*50
                trainingData.append([resizeArray, categoryIndex])
            except:
                pass
        random.shuffle(trainingData) #shuffle training data to help learn randomly instead of sequentially

        featureArray = []
        labelArray = []

        for features, label in trainingData:
            featureArray.append(features)
            labelArray.append(label)

        featureArray = np.array(featureArray).reshape(-1, imageSize, imageSize, 1) #create an np array

        pickleOut = open("features.pickle", "wb")
        pickle.dump(featureArray, pickleOut)
        pickleOut.close()

        pickleOut = open("labels.pickle", "wb")
        pickle.dump(labelArray, pickleOut)
        pickleOut.close()

def trainModel():
    features = pickle.load(open("features.pickle", "rb"))
    labels = pickle.load(open("labels.pickle", "rb"))

    features = features/255.0

    '''
    ####Single training
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.save('64*3-CNN.model')


    '''
    denseLayers = [0, 1, 2]
    layerSizes = [32, 64, 128]
    convolutionLayers = [1, 2, 3]

    ##Recursive Training
    for denseLayer in denseLayers:
        for layerSize in layerSizes:
            for convolutionLayer in convolutionLayers:
                name = "{}-c:{}-n:{}-d:{}".format(convolutionLayer, layerSize, denseLayer, int(time.time()))
                tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

                model = Sequential() #Create a sequential model

                model.add(Conv2D(layerSize, (3,3), input_shape= features.shape[1:])) #adding convolution layer with 64 layer size
                model.add(Activation("relu")) #Adding activation layer
                model.add(MaxPooling2D(pool_size=(2,2)))

                for l in range(convolutionLayer-1):
                    model.add(Conv2D(layerSize, (3,3))) #adding other convolution layer with 64 layer size
                    model.add(Activation("relu")) #other activation layer
                    model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Flatten()) #add flatten layer

                for l in range(denseLayer):
                    model.add(Dense(layerSize))
                    model.add(Activation("relu"))  # other activation layer

                model.add(Dense(1)) # Add dense layer
                model.add(Activation("sigmoid")) #add sigmoid activation layer

                model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) #compile

                model.fit(features, labels, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard]) #fit the training data

                model.save('NAME_OF_MODEL.model') #save the model

                ####TO view the training data, run "tensorboard --logdir=PATH_OF_LOG_DIR_GIVEN_IN_LINE_93"

def useTrainedModel():
    print("In trained model")
    model = tf.keras.models.load_model("NAME_OF_MODEL.model") #load the model to start prediction
    print("Prediction in progress")
    try:
        for image in os.listdir('TEST_IMAGES_DIR'):
            convertedArray = prepare('TEST_IMAGES_DIR'+image)
            if convertedArray == 'failed':
                print("Failed to predict {}".format(image))
            else:
                print(image)
                prediction = model.predict([convertedArray]) #get image from path, get its reshaped array and predict
                temp = int(prediction[0][0]) #output of prediction is array of array [[0.121]]. Read the element
                print(categories[temp]) #print the category
    except Exception as e:
        print(e)

def prepare(filePath):
    try:
        imageArray = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE) #Read filepath from
        newArray = cv2.resize(imageArray, (imageSize, imageSize)) #Resize the image to 50*50
        return newArray.reshape(-1, imageSize, imageSize, 1) #return the reshaped image as array
    except:
        return "failed" # in case of failure, return 'failed'

def checkPickle():
    try:
        with open("features.pickle", "r"):
            trainModel() # If pickle data is already present, continue with training
    except:
        createTrainingData() #else create the training data and shuffle it
        trainModel()
        useTrainedModel()


def checkModel():
    try:
        with open("NAME_OF_MODEL.model", "r"):
            print("using trained model")
            useTrainedModel()
    except:
        print("model not found")
        checkPickle()


checkModel()