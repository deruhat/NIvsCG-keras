# NIvsCG
#### Distinguishing Between Natural and Computer-Generated Images Using Convolutional Neural Networks

----------------------------------------

- This an on-going project to implement the paper in Keras.
- [Original project written in Caffe.](https://github.com/weizequan/NIvsCG)

----------------------------------------
### Splitting the Data


### Training the Model
The code for the CNN design described by the paper can be found in `nivscg.py`. Image patches used as training and validation data have to be cropped using the MPS algorithm implemented [here](https://github.com/weizequan/NIvsCG/tree/master/utils).

### Majority Voting
The code for the majority voting algorithm is in `voting.py`. A trained `.h5` model from `nivscg.py` is needed in order to run the majority voting algorithm and get the test accuracy.

## Contributers
[Abdulellah Abualshour](https://github.com/deruhat)

[Abdulmajeed Aljaloud](https://github.com/Rinzu)

## References
- W. Quan, K. Wang, D. M. Yan and X. Zhang. 2018. [Distinguishing Between Natural and Computer-Generated Images Using Convolutional Neural Networks](https://github.com/weizequan/NIvsCG). in IEEE Transactions on Information Forensics and Security, vol. 13, no. 11, pp. 2772-2787.
- W. Quan, D. M. Yan, J. Guo, W. Meng, and X. Zhang. 2016. [Maximal Poisson-disk Sampling via Sampling Radius Optimization](https://github.com/weizequan/NIvsCG/tree/master/utils). In SIGGRAPH ASIA 2016 Posters (SA '16). ACM, New York, NY, USA, Article 22, 2 pages.
