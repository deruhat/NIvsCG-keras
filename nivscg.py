from __future__ import print_function
import keras
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from time import time
from keras.callbacks import TensorBoard

import os

datagen = ImageDataGenerator()

# begin building the model
model = Sequential()

# convLayer
model.add(Conv2D(32, (7, 7), input_shape=(233, 233, 3)))

# C1
model.add(Conv2D(64, (7, 7), input_shape=(227, 227, 32)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# C2
model.add(Conv2D(48, (5, 5), input_shape=(55, 55, 64)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# C3
model.add(Conv2D(64, (3, 3), input_shape=(25, 25, 48)))
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
model.add(Activation('softmax'))

# optimizer
sgd = optimizers.SGD(lr=0.001)

# loss function is binary crossentropy (goof for binary classification)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# prepare for training
batch_size = 128

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

# tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# start training
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=34,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[tensorboard])
model.save_weights('NIvsCG_model.h5')