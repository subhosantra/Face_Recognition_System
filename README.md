# Face_Recognition_System

A Face Recognition System for 2.5 lakh unique faces using a multi-level KNN clusters and a ResNet-34 model.Recognition involves
pre-processing (face detection & landmark generation + alignment) and then inference using ResNet-34 model. The KNN structure
involve 4 layers and each layer node have maximum of 9 levels. The final output is a HashMap with keys of type int and value is
array of normalized vector for future recognition purpose.


File Description:
1. Aligner_frontal_v_1.0 --> Used for the frontal face detection and creation of embedding of the images using transfer learning from a ResNet-34 model.(We can obtain image embeddings from a ResNet-34 by taking the output of its second last Fully-connected layer which has a dimension of 512.)

2. Knn_model_v_1.0 --> Used for the classification of all the data using K-mean algorithm and creation of 1-D Numpy-array containg centroid of ech layer.

3. Prediction_v_1.0 --> Used to predict the label of new input image using score of recognition which is basically a dot product divide by multiplication of magnitude of two vector.
