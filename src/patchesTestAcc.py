
from __future__ import print_function
import keras
from keras import backend as K
from keras.preprocessing.image import *
from keras.models import Sequential, load_model

model = load_model('../models/NIvsCG_model_100epochs_None-NoneStep.h5')
batch_size = 32
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        '../../datasets/patches/test',
        target_size=(233, 233),
        batch_size=batch_size,
        class_mode='binary')

score = model.evaluate_generator(test_generator, verbose=1)
print(score)
'''
score: [0.8123159432858229, 0.85934375]
'''