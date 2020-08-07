# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:32:37 2020

@author: jankit
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import time
import json

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

def run_model(path,num_layers,num_clusters):
    
    #load inference of the training data
    face_embeddings = np.load(path)
    face_embeddings = np.reshape(face_embeddings,(int(face_embeddings.shape[0]/512),512))
    
    #convert  the face_embedding  to panda dat frame nd add index column on the rightmst side
    df = pd.DataFrame(face_embeddings)
    ind=[]    
    for i in range(df.shape[0]) :
        ind.append(i)
    df["index"]=ind
    
    #Cluster and make a centroids_1D which contain all centroid in 1-D pattern
    num_samples=df.shape[0]
    centroids_1D=[]
    centroids_1D = np.asarray(centroids_1D)
    
    #forming the layers having cluster value   
    for i in range(num_layers):
        print("****Layer{}***".format(i))
        start = time.time()
        if i==0:
            pos=[0,num_samples]
        else:
            
            #finding the partition index on the basis of previous layer cluster prediction
            pos=[0]
            curr=0
            temp= np.asarray(df)
            for index in range(num_samples):
                if temp[index,-1]!=curr:
                    pos.append(index)
                    curr=temp[index,-1]
            pos.append(num_samples)
            
        #seperating the all the element of particular cluster and doing the prdeiction            
        df_arr=np.asarray(df)
        temp_y = []
        temp_y = np.asarray(temp_y)
        for j in range(len(pos)-1):
            temp_df=df_arr[pos[j]:pos[j+1],:]
            if(i == 0):
                X = temp_df[:,:-1]
            else:
                X = temp_df[:,:-(i+1)] 
                
            #prediction and making centroids_1D consisting of centroid
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1000)
            temp_y = np.append(temp_y,kmeans.fit_predict(X))
            centroids = kmeans.cluster_centers_
            centroids_1D = np.append(centroids_1D,centroids)
            
        #sorting the dataframe with respect to the all the layers available to get particular order cluster value    
        df["layer{}".format(i)] = temp_y    
        list_layers = ["layer{}".format(k) for k in range(i+1)]
        df = df.sort_values(by=list_layers)
        end = time.time()
        print("[INFO] Clustering took {:.5} seconds".format(end - start))
    
    #Convert centroid_1D to Hash_map   
    maps= centroids_1D
    maps = np.reshape(maps,(int(maps.shape[0]/512),512))
    hash_map={}
    for i in range(maps.shape[0]):
        hash_map[int(converter(i,num_clusters))]=maps[i,:].tolist()
    
    #creation of Hash_map json file        
    with open('E:\\A_synaptic_sense\\faces_umd\\hash_map.json', 'w') as fp:
      json.dump(hash_map, fp, sort_keys=True, indent=4)
    print("[INFO]: Hash_Map Json File Created") 
    
    #creation of cluster tree csv file  
    df_arr=np.asarray(df)
    np.save("E:\\A_synaptic_sense\\faces_umd\\cluster_tree.npy",df_arr)
    print("[INFO]: Cluster Tree csv file created")


if __name__ == '__main__':
    path="E:\\A_synaptic_sense\\faces_umd\\inference.npy"
    num_layers=4
    num_clusters=4
    run_model(path,num_layers,num_clusters)
