# NIvsCG
#### Distinguishing Between Natural and Computer-Generated Images Using Convolutional Neural Networks
 
----------------------------------------

- This an on-going project to implement the paper in Keras.
- [Original project written in Caffe.](https://github.com/weizequan/NIvsCG)

----------------------------------------
### Preparing the Workspace
Make sure you have the project structured as follows:
```
├── checkpoints
├── logs
├── models
├── results
├── src
   ├── model.py
   ├── voting.py
   ├── patchesTestAcc.py
├── utils
   ├── mps
   ├── imageMpsCrop.m
   ├── makePatches.m
   └── imageNamesFileMaker.py
└── datasets
   ├── full
       ├── personal
           ├── 000001.jpg 
           └── ...
       └── prcg
           ├── 000001.jpg 
           └── ...
   └── patches
       ├── train
           ├── personal
              ├── patch-001.bmp 
              └── ...
           └── prcg
              ├── patch-001.bmp 
              └── ...
       ├── valid
           ├── personal
              ├── patch-001.bmp 
              └── ...
           └── prcg
              ├── patch-001.bmp 
              └── ...
       ├── test
           ├── personal
              ├── patch-001.bmp 
              └── ...
           └── prcg
              ├── patch-001.bmp 
              └── ...
       └── test-majority-voting
           ├── all
              ├── patch-001.bmp 
              └── ...
           └── filenames.txt
```
### Dataset
We used the Personal and PRCG datasets for our two classes.
- Google and PRCG datasets can be downloaded [here](https://drive.google.com/file/d/0BwHrTmiHZpQCRFFVMWFEaWsyaWM/view).
- Personal dataset can be downloaded [here](http://www.ee.columbia.edu/~dvmmweb/dvmm/downloads/PIM_PRCG_dataset/personal_columbia_downsized_jpeg.zip).

### Training the Model
The code for the CNN design described by the paper can be found in `model.py`. Image patches used as training and validation data have to be cropped using the MPS algorithm implemented [here](https://github.com/weizequan/NIvsCG/tree/master/utils).

### Majority Voting
The code for the majority voting algorithm is in `voting.py`. A trained `.h5` model from `model.py` is needed in order to run the majority voting algorithm and get the test accuracy.

### Results
So far we're able to achieve ~~95%~~ ~~96.88%~~ 97.18% accuracy on image classification. We take 200 patches that cover the whole image from each test sample using the MPS algorithm, and take the majority vote of these 200 patches to decide that test image's class.

You can download the trained model [here](https://drive.google.com/file/d/1YLQ7duhInAyHfUrbGUJttlmduB0D0nve/view?usp=sharing).

## Contributers
[Abdulellah Abualshour](https://github.com/deruhat)

[Abdulmajeed Aljaloud](https://github.com/Rinzu)

## References
- W. Quan, K. Wang, D. M. Yan and X. Zhang. 2018. [Distinguishing Between Natural and Computer-Generated Images Using Convolutional Neural Networks](https://github.com/weizequan/NIvsCG). In IEEE Transactions on Information Forensics and Security, vol. 13, no. 11, pp. 2772-2787.
- W. Quan, D. M. Yan, J. Guo, W. Meng, and X. Zhang. 2016. [Maximal Poisson-disk Sampling via Sampling Radius Optimization](https://github.com/weizequan/NIvsCG/tree/master/utils). In SIGGRAPH ASIA 2016 Posters (SA '16). ACM, New York, NY, USA, Article 22, 2 pages.
