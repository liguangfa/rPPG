# Remote heart rate estimation based on rPPG amplification

#-------------------------

In this article, we propose an encoder-decoder model (ED-RPPG) to amplify the faint skin color changes of the face, so as to improve the signal-to-noise ratio of the target signal, thereby achieving a higher-precision estimation of the BVP signal and heart rate. In addition, we designed a rPPG video generation algorithm (RVG), which can synthesize facial videos containing rPPG information and the corresponding spatiotemporal BVP labels. The proposed method has achieved superior performance on the public-domain PURE and UBFC-RPPG databases.

#---------------------

The encoder-decoer.py file is a model file. You need to train the model with file train.py to obtain the network weights before using it.
After the pre-training is completed, use the test.py file to test the real data.
