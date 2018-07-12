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
from keras.preprocessing.image import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, regularizers
from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

import os

# subclassing TensorBoard to show LR
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

datagen = ImageDataGenerator()

# begin building the model
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
model.add(Dense(1))
model.add(Activation('sigmoid'))

# optimizer
adam = optimizers.Adam(lr=1e-10)

# loss function is binary crossentropy (for binary classification)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# prepare for training
batch_size = 32

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'utils/output-data/train', 
        target_size=(233, 233),  # patch size 
        batch_size=batch_size,
        class_mode='binary')  # binary_crossentropy loss

validation_generator = test_datagen.flow_from_directory(
        'utils/output-data/valid',
        target_size=(233, 233),
        batch_size=batch_size,
        class_mode='binary')

# callbacks
tensorboard = LRTensorBoard(log_dir="logs/{}".format(time()))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0)
checkpoint = ModelCheckpoint("checkpoints/weights_regularization_added.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=5)

# load trained model, remove this line if training from scratch
model.load_weights('NIvsCG_weights.h5')

# start training
model.fit_generator(
        train_generator,
        steps_per_epoch=None,
        epochs=250,
        validation_data=validation_generator,
        validation_steps=None,
        callbacks=[tensorboard, reduce_lr, checkpoint])

model.save_weights('NIvsCG_weights.h5') 