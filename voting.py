##########################################################################
#                                                                        #
#    Implementation of Distinguishing Between Natural and Computer-      #
#        Generated Images Using Convolutional Neural Networks            #
#                               (NIvsCG)                                 #
#               Majority Voting & Final Test Accuracies                  #
#                                                                        #
##########################################################################

from __future__ import print_function
import numpy as np
import sys, os
import keras
from keras.preprocessing.image import *
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, regularizers
from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

# Load model
model = load_model('NIvsCG_model.h5')

# optimizer
adam = optimizers.Adam(lr=1e-8)

# loss function is binary crossentropy (for binary classification)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

