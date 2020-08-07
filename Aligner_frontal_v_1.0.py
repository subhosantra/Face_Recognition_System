# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:12:42 2020

@author: jankit
"""

from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils.face_utils import rect_to_bb
import os
import dlib
import cv2
import numpy as np
import time


model_path = 'E:\\A_synaptic_sense\\Summer Internship\\mxnet_exported_res34.onnx'
resnet = cv2.dnn.readNetFromONNX(model_path)

def infe(path):
        img_data = cv2.imread(path)
        blob = cv2.dnn.blobFromImage(img_data, scalefactor=1.0, size=(112, 112), mean=(0, 0, 0),swapRB=True, crop=False)
        resnet.setInput(blob)
        preds = resnet.forward()
        return preds[0]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:\\A_synaptic_sense\\Alignment\\facial-landmarks\\shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=112)

def aparts(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 2)
    
    if rects:
        rect =rects[0]
        
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)
        gray1= cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
        rut=detector(gray1,1)
        if rut:
            shape = predictor(gray1,rut[0])
            shape = face_utils.shape_to_np(shape)
            
            leftEyePts = shape[36:42]
            rightEyePts=shape[42:48]
            
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
            
            mid_x=int((leftEyeCenter[0]+rightEyeCenter[0])/2);
            #mid_y=int((leftEyeCenter[1]+rightEyeCenter[1])/2);
            apart=abs(mid_x-shape[30][0])
        else:
            return 1000000
         
            
        return apart 
    
    else:
        return 1000000
    
    
if __name__ == '__main__':
  
#frontal detection and saving it to frontal face folder    
    idx=1
    i=0
    while True:
        dist={}
        k=1
        while True:
            path= "E:\\A_synaptic_sense\\faces_umd\\emore_images_2\\"+str(i).zfill(6)+"\\"+str(idx).zfill(8) +".jpg"  #checking all images of same person
            print (path)
            if os.path.exists(path):
                if(k==1):
                    try:
                      dist[idx]=aparts(path)
                    except:
                      idx=idx+1
                      continue
                if(k==1 and dist[idx]<= 5):  #  5 is the threshold value of x-axis for the difference between mid point of eyes and the tip of the nose
                    k=0
                #if k==1 we will call the apart function .once we find value lesser then threshold we dont need to check further hence k become 0    
                idx=idx+1
                print(idx)
            else:
                break;

        temp=min(dist.values())
        res = [key for key in dist if dist[key] == temp]
        path= "E:\\A_synaptic_sense\\faces_umd\\emore_images_2\\"+str(i).zfill(6)+"\\"+str(res[0]).zfill(8) +".jpg"  #Align the face of most frontal image and saving it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        faceAligned = fa.align(image, gray, rects[0])

        label_path="E:\\A_synaptic_sense\\faces_umd\\"
        save_path = os.path.join(label_path, 'frontal_faces')
        if not os.path.exists(save_path):
         os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, str(i).zfill(6) + '.jpg'), faceAligned)
        i=i+1
        path= "E:\\A_synaptic_sense\\faces_umd\\emore_images_2\\"+str(i).zfill(6)+"\\"+str(idx).zfill(8) +".jpg"
        if not os.path.exists(path):
            break

#Create Inferenced of Frontal Image     
        start=time.time()
        i=0
        inferenced=[]
        
        while True:
            path= "E:\\A_synaptic_sense\\faces_umd\\frontal_face\\"+str(i).zfill(6)+".jpg"
            if not os.path.exists(path):
                break
            inferenced=np.append(inferenced,infe(path))
            i=i+1
            print (i)
        end=time.time()
        print("INFO[] Creating Inference file took {:0.5} seconds".format(end - start))
        save_path="E:\\A_synaptic_sense\\faces_umd\\inference.npy"
    
        np.save(save_path,inferenced)
        
    
