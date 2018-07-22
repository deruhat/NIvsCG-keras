# NIvsCG
#### Distinguishing Between Natural and Computer-Generated Images Using Convolutional Neural Networks

----------------------------------------

- This an on-going project to implement the paper in Keras.
- Original project written in Caffe: https://github.com/weizequan/NIvsCG

----------------------------------------

### Training the model:
The code for the CNN design described by the paper can be found in `nivscg.py`. Image patches used as training and validation data have to be cropped using the MPS algorithm implemented here: https://github.com/weizequan/NIvsCG

### Majority Voting:
The code for the majority voting algorithm is in `voting.py`. A trained `.h5` model from `nivscg.py` is needed in order to run the majority voting algorithm and get the test accuracy.

## Contributers:
Abdulellah Abualshour | https://github.com/deruhat
Abdulmajeed Aljaloud | https://github.com/Rinzu
