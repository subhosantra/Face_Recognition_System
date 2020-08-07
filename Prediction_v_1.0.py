# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:50:51 2020

@author: jankit
"""

import numpy as np
import time
import math
import cv2
import json
import matplotlib.pyplot as plt
import os

def converter (n,num_clusters):
    if n == 0:
        return '1'
    nums = []
    k=0
    while n:
        if (n<num_clusters and k==0):
            r=(n%num_clusters)+1
            n=int(n/num_clusters)
        else:    
            n, r = divmod(n, num_clusters)
            #print(n)
            if(n/num_clusters>0):
                r+=1   
            k=1
            
        nums.append(str(r))   
    return ''.join(reversed(nums))

def get_divs(df_arr,layer,num_layer):
    pos=[0]
    curr=0
    num_samples=df_arr.shape[0]
    #print(num_samples)
    for index in range(num_samples):
        if df_arr[index,(-num_layer)+layer]!=curr:
            pos.append(index)
            curr=df_arr[index,(-num_layer)+layer]
    pos.append(num_samples)
    return pos

def score(vec1,vec2):
    score=0
    vec1_l2=0
    vec2_l2=0

    for i in range(512):
        vec1_l2=vec1_l2+(vec1[i]*vec1[i])
        vec2_l2=vec2_l2+(vec2[i]*vec2[i])
        score=np.dot(vec1,vec2)

    vec1_l2=math.sqrt(vec1_l2)
    vec2_l2=math.sqrt(vec2_l2)  
    #print(vec1_l2*vec2_l2)
    score=score/(vec1_l2*vec2_l2)

    return score  

def pre(df_arr,image_path,model,num_layer,num_cluster):
    start=time.time()
    
    #loading hash_map.json file which contain th ecentroids of node in heap array style starting from index 1    
    with open('E:\\A_synaptic_sense\\faces_umd\\hash_map.json','r') as fp:
      hash_map=json.load(fp)

    #Converting the input image to (512 ,1) dimension vector using resnet_model
    img_data = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img_data, scalefactor=1.0, size=(112, 112), mean=(0, 0, 0),swapRB=True, crop=False)
    model.setInput(blob)
    embedding = model.forward()
    embedding =np.reshape(embedding,(512,1))
 
    #Find and store all the node through which we will traverse to last node   
    maxi=-10000000
    index=0
    index1=0
    save=0
    ind=[]
    for j in range(num_layer):
        for i in range((num_cluster*index1)+1,(num_cluster*index1)+(num_cluster+1)):
            distan=score(hash_map[str(int(converter(i-1,num_cluster)))],embedding)
            if maxi<distan:
                maxi=distan
                index=i-1
        
        save=(index-((num_cluster*index1)))
        index1=index+1
        ind=np.append(ind,save)
        maxi=-10000000
        

    #Seperating the all the data for which embedding gives highest score upto thr last layer centroid
    print("Traversal List for the Current input: {}".format(ind))    
    for i in range(num_layer):
        layer_=get_divs(df_arr,i,num_layer)
        lat=layer_[int(ind[i]):(int(ind[i])+2)]
        df_arr=df_arr[lat[0]:lat[1],:]
    

    #Calculating the score of every image in extracted dataset with the input image 
    maxi=-10000000
    index=0
    score_card=[]
    for i in range(df_arr.shape[0]):
            scored=score(df_arr[i,:-(num_layer+1)],embedding)
            score_card.append(scored)
            if maxi<scored:
                maxi=scored
                index=i
    end=time.time()     
    print("[INFO]: Time taken for finding Output is {:.5} seconds".format(end - start)) 

           
    #plot the the score of extracted dataset with the input image               
    plt.plot(score_card)
    plt.ylabel("score")
    plt.xlabel("images in cluster")
    plt.show()
    
 
    #Display and print the predicted image and label   
    img_label=int(df_arr[index][512])
    image = cv2.imread("E:\\A_synaptic_sense\\faces_umd\\frontal_face\\"+str(img_label).zfill(6)+".jpg")    
    cv2.imshow("Predicted Image",image)
    cv2.waitKey(0) 
    return img_label        


if __name__ == '__main__':
    model_path = 'E:\\A_synaptic_sense\\Summer Internship\\mxnet_exported_res34.onnx'
    resnet = cv2.dnn.readNetFromONNX(model_path)
    df_arr=np.load("E:\\A_synaptic_sense\\faces_umd\\cluster_tree.npy")
    img_path="E:\\A_synaptic_sense\\faces_umd\\frontal_face\\000786.jpg"
    num_layer=4
    num_cluster=4
    s=pre(df_arr,img_path,resnet,num_layer,num_cluster)
    print (s)
    
    
'''   
#testing accuracy
    idx=1
    i=0
    while True:
        dist=0
        m=0
        k=1
        while True:
            path= "E:\\A_synaptic_sense\\faces_umd\\emore_images_2\\"+str(i).zfill(6)+"\\"+str(idx).zfill(8) +".jpg"
            if os.path.exists(path):
                s=pre(df_arr,path,resnet,num_layer,num_cluster)
                if(s==i):
                  dist=dist+1
                idx=idx+1
                m=m+1
                print(idx)

            else:
                break;
        if(m!=0):
         acc=dist/m
        else:
         acc=0
        dist=0
        m=0 
        print("accuracy for {} :".format(str(i).zfill(6)))
        print(acc)
        i=i+1
        path= "E:\\A_synaptic_sense\\faces_umd\\emore_images_2\\"+str(i).zfill(6)+"\\"+str(idx).zfill(8) +".jpg"
        if not os.path.exists(path):
            break
'''
