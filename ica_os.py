import numpy as np
import os

import cv2
w = 307
h=307
for i in range(100):
  print(i)
  np.save('os_i.npy',i)
  os.system("python ./ICA_hyper.py")
  if i == 0:
    acc_plot =  np.load('current_results.npy')

  else:
    acc_plot =  acc_plot + np.load('current_results.npy')


     
  show = acc_plot.copy()
  for endm in range(5):
     show[:,endm]= (show[:,endm]-show[:,endm].min())/(show[:,endm].max()-show[:,endm].min())
  print(show.shape)
  np.save('plots/accumulate_5.npy',show)
  index_list = [0,1,2,3,4]#np.load('index_list.npy')
  

  save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h)),-1)

  
  
  save_img = (save_img*255).astype(np.uint8)
  cv2.imwrite('plots/accumulate_5.png',save_img)
  
  

