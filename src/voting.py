##########################################################################
#                                                                        #
#    Implementation of Distinguishing Between Natural and Computer-      #
#        Generated Images Using Convolutional Neural Networks            #
#                               (NIvsCG)                                 #
#               Majority Voting & Final Test Accuracies                  #
#                                                                        #
##########################################################################

# Computing average accuracy on cropped patch (240 x 240) and full-sized image after voting
# This file can also be modified for other patch sizes, i.e., 180 x 180, 120 x 120, etc. 

from __future__ import print_function
import keras
from keras import backend as K
from keras.preprocessing.image import *
from keras.models import Sequential, load_model
from PIL import Image
import numpy as np

def load_image(fname) :
    img = Image.open(fname)
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = data[0:233,0:233,:]
    return data.reshape((1,) + data.shape)

model = load_model('../checkpoints/model_2/model.08-0.67.h5')
test_dir = "../datasets/patches/test-majority_voting/"
kPrcgNum = 160

imageLabel = []
testLabel = []
imageTmp = []
testTmp = []
oriImageLabel = []  # one dimension list
oriTestLabel = []  # one dimension list

# test240_30_num.txt records the name, label of image patch and the number of cropped patches for each test image (i.e., 30)
# for example:
# prcg_images/set1-arch-11-1.bmp(the name of 1-th patch) 0(label)
# prcg_images/set1-arch-11-2.bmp(the name of 2-th patch) 0(label)
# ...
# prcg_images/set1-arch-11-30.bmp(the name of 30-th patch) 0(label)
# 30 (the number of cropped patches for each test image)
# ...

# Note that, [1] and [2] need to be refined for your own data
testImageDir = test_dir + 'filenames.txt'  # [1]the info of test image patch
testImageFile = open(testImageDir, 'r')
for line in testImageFile:
    twoTuple = line.split()
    if len(twoTuple) == 2:
        image = load_image(test_dir + 'all/' + twoTuple[0])  # [2]the test image dir
        imageTmp.append(int(twoTuple[1]))
        output = model.predict([image], batch_size = 1)
        output_prob = output[0]
        testTmp.append(output_prob.argmax())
    else:
        oriImageLabel.extend(imageTmp)
        oriTestLabel.extend(testTmp)
        imageLabel.append(imageTmp)
        testLabel.append(testTmp)
        imageTmp = []
        testTmp = []

testImageFile.close()

imageCropNum = [len(x) for x in imageLabel]
imageCropNumNp = np.array(imageCropNum)
imageLabelNp = np.array(imageLabel)
testLabelNp = np.array(testLabel)

#  Computing average accuracy on patches
result = np.array(oriImageLabel) == np.array(oriTestLabel)

prcg_result = result[:kPrcgNum*200]
google_result = result[kPrcgNum*200:]
print('The number of patches: %d (%d PCRG, %d personal)' % (len(oriImageLabel), len(prcg_result), len(google_result)))
print('Accuracy on Patches:')
print('-The personal (NI) accuracy is:', google_result.sum()*1.0/len(google_result))
print('-The prcg (CG) accuracy is:', prcg_result.sum()*1.0/len(prcg_result))
print('-CG patches misclassified as natural patches (CGmcNI) is:', (len(prcg_result) - prcg_result.sum())*1.0/len(prcg_result))
print('-natural patches misclassified as CG patches (NImcCG) is:', (len(google_result) - google_result.sum())*1.0/len(google_result))
print('-The average accuracy is:', result.sum()*1.0/len(result))

#  Computing average accuracy on full-sized images (29 patches and majority voting)
result = np.arange(len(imageLabel))
for x in range(len(imageLabel)):
    tmp = np.array(imageLabelNp[x]) == np.array(testLabelNp[x])
    result[x] = np.sum(tmp[:-1]) > imageCropNumNp[x]//2 - 1

prcg_result = result[:kPrcgNum]
google_result = result[kPrcgNum:]
print('\nThe number of full-sized testing images is: %d (%d PCRG, %d personal)' % (len(imageLabel), len(prcg_result), len(google_result)))
print('The average accuracy on full-sized images after majority voting:')
print('-The personal (NI) accuracy is:', google_result.sum()*1.0/len(google_result))
print('-The prcg (CG) accuracy is:', prcg_result.sum()*1.0/len(prcg_result))
print('-CG images misclassified as natural images (CGmcNI) is:', (len(prcg_result) - prcg_result.sum())*1.0/len(prcg_result))
print('-natural images misclassified as CG images (NImcCG) is:', (len(google_result) - google_result.sum())*1.0/len(google_result))
print('-The average accuracy is:', result.sum()*1.0/len(result))
