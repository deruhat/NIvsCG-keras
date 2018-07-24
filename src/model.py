##########################################################################
#                                                                        #
#    Implementation of Distinguishing Between Natural and Computer-      #
#        Generated Images Using Convolutional Neural Networks            #
#                               (NIvsCG)                                 #
#                             Model Design                               #
#                                                                        #
##########################################################################

from __future__ import print_function
import keras
from keras import backend as K
from keras.preprocessing.image import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, regularizers
from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import multi_gpu_model

import os

# subclassing TensorBoard to show LR
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

# building the model
model = Sequential()

# convLayer
model.add(Conv2D(32, (7, 7), input_shape=(233, 233, 3), kernel_regularizer=regularizers.l1(0.01)))

# C1
model.add(Conv2D(64, (7, 7)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# C2
model.add(Conv2D(48, (5, 5)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# C3
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# FC4 (Dense)
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# FC5
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(2, activation='softmax'))

# optimizer
adam = optimizers.Adam(lr=1e-5)

# multi-gpu
model = multi_gpu_model(model, gpus=4)

# loss function is binary crossentropy (for binary classification)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# make the data generators for image data
batch_size = 128

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '../datasets/patches/train', 
        target_size=(233, 233),  # input size as in paper
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '../datasets/patches/valid',
        target_size=(233, 233),
        batch_size=batch_size,
        class_mode='binary')

# load trained model
# model = load_model('../models/NIvsCG_model_20epochs_None-NoneStep.h5')

# make callbacks to use while training
tensorboard = LRTensorBoard(log_dir="../logs/{}".format(time()))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0)
# make sure the path to the directory inside 'checkpoints' exists to avoid failure while saving:
checkpoint = ModelCheckpoint("../checkpoints/model_2/model.{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=4)

# start training
model.fit_generator(
        train_generator,
        steps_per_epoch=None,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=None,
        callbacks=[tensorboard, reduce_lr, checkpoint])

# save the model if ever finish
model.save('../models/NIvsCG_model_100epochs_None-NoneStep.h5') 
