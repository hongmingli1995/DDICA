import numpy as np
import os

import cv2


run_i = np.load('run_i.npy')
w = 307
h=307

endnum = 8
k = 0
for i in range(50):
  print(i)
  np.save('os_i.npy',i)
  os.system("python ./ICA_imge_raw.py")
  
  current_loss = np.load('current_loss.npy')
  if 1:#current_loss<=3.5:
    
    if k == 0:
      acc_plot =  np.load('current_results.npy')
  
  
  
    else:
      scores = np.ones([endnum,endnum])#np.diag(5)
      current_plot = np.load('current_results.npy')
      acc_plot_norm = acc_plot.copy()
      
      #asa = np.max(acc_plot_norm,-1)
      #aaa= acc_plot_norm - np.repeat(np.expand_dims(asa,-1),5,-1)
      #aaa[aaa ==0] =1
      #aaa[aaa<=0] = 0
      #acc_plot_norm = aaa
      
      for endm in range(endnum):
        acc_plot_norm[:,endm]= (acc_plot_norm[:,endm]-acc_plot_norm[:,endm].min())/(acc_plot_norm[:,endm].max()-acc_plot_norm[:,endm].min()+0.0001)
  
      for itruth in range(endnum):
        for ishow in range(endnum):
          single_truth = acc_plot_norm[:,itruth]
          single_show = current_plot[:,ishow]
                          
          scores[itruth,ishow] = np.sum(np.abs(single_truth - single_show))/len(single_show)
      
      
  
      for endm in range(endnum):
        row = np.where(scores == scores.min())[0][0] 
        col = np.where(scores == scores.min())[1][0] 
        acc_plot[:,row] = acc_plot[:,row]+current_plot[:,col]
        scores[row,:] = 1
        scores[:,col] = 1
    
  
  
       
    show = acc_plot.copy()
    for endm in range(endnum):
       show[:,endm]= (show[:,endm]-show[:,endm].min())/(show[:,endm].max()-show[:,endm].min()+0.0001)
    print(show.shape)
    np.save('plots/acu_no_1.npy',show)
    index_list = [0,1,2,3,4,5,6,7]#np.load('index_list.npy')
    
  
    #save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:#,index_list[4]].reshape(w,h)),-1)
    
    if endnum == 8:
      save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h),show[:,index_list[5]].reshape(w,h),show[:,index_list[6]].reshape(w,h),show[:,index_list[7]].reshape(w,h)),-1)
    else:
      save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h)),-1)
    
  
    
    
    save_img = (save_img*255).astype(np.uint8)
    cv2.imwrite('plots/accumulate_'+str(run_i)+'_raw.png',save_img)
    k+=1