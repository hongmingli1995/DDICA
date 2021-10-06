# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:02:54 2021

@author: 61995
"""

# import numpy as np
# img_list =[]
# for i in range(1):
#     for j in range(100):
#         img = np.load('results8_'+str(j)+'_'+str(i)+'.npy')
#         for k in range(8):
#             img_list.append(img[:,k].reshape([307,307]))
# img_list = np.array(img_list)

# np.save('img_list_8.npy',img_list)

import matplotlib.pyplot as plt
import numpy as np
import cv2
img_list = np.load('img_list_8.npy')

X_train = img_list.reshape(len(img_list),-1)

from sklearn.cluster import MiniBatchKMeans,KMeans
#from equal_groups import EqualGroupsKMeans
kmeans = KMeans(n_clusters = 5, n_init =20)

kmeans.fit(X_train)

#%%
labels = kmeans.labels_
# img_con = []
# for i in range(5):
#     img = np.zeros([307,307])
#     for j in range(1500):
#         if labels[j] == i:
#             img = img + X_train[j].reshape([307,307])
#     img = (img - img.min()) / (img.max() - img.min())
#     img_con.append(img) 
# img_con = np. array(img_con)
center = kmeans.cluster_centers_.reshape([-1,307,307])
img_con = center
table = []
for i in range(800):
    img = X_train[i].reshape([307,307])
    dis = []   
    for j in range(5):
        dis.append(((img-center[j])**2).sum())
    table.append(dis)
table = np.array(table)

# #

img_con = []
for i in range(5):
    img = np.zeros([307,307])
    #imgs = X_train[np.argsort(table[:,0])[0:300]].reshape([-1,307,307])
    img_con.append(X_train[np.argsort(table[:,i])[0:100]].reshape([-1,307,307]).sum(0))
    
img_con = np. array(img_con)    
    
    

for i in range(5):
    img_con[i] = img_con[i] = (img_con[i] - img_con[i].min()) / (img_con[i].max() - img_con[i].min())
    #img_con[i][img_con[i]<=0.8] = 0

# show = np.max(img_con,0)
# show2= img_con - np.repeat(np.expand_dims(show,0),5,0)
# show2[show2==0] = 1
# show2[show2<0] = 0
#img_con = show2
plt.imshow(np.concatenate((img_con[0],img_con[1],img_con[2],img_con[3],img_con[4]),-1))
plt.show()