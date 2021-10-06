# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:07:20 2021

@author: 61995
"""
from scipy.io import loadmat
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy import signal
from image import *
import cv2

os_i = np.load('os_i.npy')
source = loadmat('data_urban/Urban_R162.mat')
truth = loadmat('data_urban/end5_groundTruth.mat')
source_data = source['Y']
X_all = source_data.T/255
truth_data = truth['A']
truth_img = truth_data.T
if 0:#os_i!= 0:
  truth_img = np.load('plots/accumulate_5.npy')

asa = np.max(truth_img,-1)
aaa= truth_img - np.repeat(np.expand_dims(asa,-1),5,-1)
aaa[aaa ==0] =1
aaa[aaa<=0] = 0
truth_img = aaa



# for i in range(len(truth_data)):
#     plt.imshow(truth_data[i].reshape([307,307]))
#     plt.show()
w = 307
h=307
ica = FastICA(n_components=5,max_iter=200, tol=0.001)
S_ = ica.fit_transform(X_all)

#S_[:,0]= (S_[:,0]-S_[:,0].min())/(S_[:,0].max()-S_[:,0].min())#(show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
#S_[:,1]= (S_[:,1]-S_[:,1].min())/(S_[:,1].max()-S_[:,1].min())#(show_si[:,1]-show_si[:,1].mean())/show_si[:,1].std()
#S_[:,2]= (S_[:,2]-S_[:,2].min())/(S_[:,2].max()-S_[:,2].min())
#S_[:,3]= (S_[:,3]-S_[:,3].min())/(S_[:,3].max()-S_[:,3].min())
#S_[:,4]= (S_[:,4]-S_[:,4].min())/(S_[:,4].max()-S_[:,4].min())

#plt.imshow(np.concatenate((S_[:,0].reshape(w,h),S_[:,1].reshape(w,h),S_[:,2].reshape(w,h),S_[:,3].reshape(w,h),S_[:,4].reshape(w,h)
#                            ),-1))
#plt.show()


#%%
    
def GaussianMatrix(X,sigma):
    G = torch.mm(X, X.T)
    K = 2*G-(torch.diag(G).reshape([1,G.size()[0]]))
    K = 1/(2*sigma**2)*(K-(torch.diag(G).reshape([G.size()[0],1])))
    K = torch.exp(K)

    return K


def MMI8(variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,sigma1,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma1)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma1)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    input4 = variable4
    K_4 = GaussianMatrix(input4,sigma1)/(input4.size(dim=0))
    L_4,_ = torch.symeig(K_4,eigenvectors=True)
    lambda_4 = torch.abs(L_4)
    #lambda_y = L_y
    H_4 = (1/(1-alpha))*torch.log2((torch.sum(lambda_4 ** alpha)))
    
    input5 = variable5
    K_5 = GaussianMatrix(input5,sigma1)/(input5.size(dim=0))
    L_5,_ = torch.symeig(K_5,eigenvectors=True)
    lambda_5 = torch.abs(L_5)
    #lambda_y = L_y
    H_5 = (1/(1-alpha))*torch.log2((torch.sum(lambda_5 ** alpha)))
    
    input6 = variable6
    K_6 = GaussianMatrix(input6,sigma1)/(input6.size(dim=0))
    L_6,_ = torch.symeig(K_6,eigenvectors=True)
    lambda_6 = torch.abs(L_6)
    #lambda_y = L_y
    H_6 = (1/(1-alpha))*torch.log2((torch.sum(lambda_6 ** alpha)))
    
    input7 = variable7
    K_7 = GaussianMatrix(input7,sigma1)/(input7.size(dim=0))
    L_7,_ = torch.symeig(K_7,eigenvectors=True)
    lambda_7 = torch.abs(L_7)
    #lambda_y = L_y
    H_7 = (1/(1-alpha))*torch.log2((torch.sum(lambda_7 ** alpha)))
    
    input8 = variable8
    K_8 = GaussianMatrix(input8,sigma1)/(input8.size(dim=0))
    L_8,_ = torch.symeig(K_8,eigenvectors=True)
    lambda_8 = torch.abs(L_8)
    #lambda_y = L_y
    H_8 = (1/(1-alpha))*torch.log2((torch.sum(lambda_8 ** alpha)))
    

    
    K_xyz = K_x*K_y*K_z*K_4*K_5*K_6*K_7*K_8*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_x+H_y+H_z+H_4+H_5+H_6+H_7+H_8 - H_xyz



def MMI5(variable1,variable2,variable3,variable4,variable5,sigma1,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma1)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma1)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    input4 = variable4
    K_4 = GaussianMatrix(input4,sigma1)/(input4.size(dim=0))
    L_4,_ = torch.symeig(K_4,eigenvectors=True)
    lambda_4 = torch.abs(L_4)
    #lambda_y = L_y
    H_4 = (1/(1-alpha))*torch.log2((torch.sum(lambda_4 ** alpha)))
    
    input5 = variable5
    K_5 = GaussianMatrix(input5,sigma1)/(input5.size(dim=0))
    L_5,_ = torch.symeig(K_5,eigenvectors=True)
    lambda_5 = torch.abs(L_5)
    #lambda_y = L_y
    H_5 = (1/(1-alpha))*torch.log2((torch.sum(lambda_5 ** alpha)))
    

    

    
    K_xyz = K_x*K_y*K_z*K_4*K_5*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_x+H_y+H_z+H_4+H_5 - H_xyz

# def MMI(variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,sigma1,alpha):
#     input1 = variable1
#     K_1 = GaussianMatrix(input1,sigma)/(input1.size(dim=0))
#     L_1,_ = torch.symeig(K_1,eigenvectors=True)
#     lambda_1 = torch.abs(L_1)
#     H_1 = (1/(1-alpha))*torch.log((torch.sum(lambda_1 ** alpha)))
    
    
#     input2 = variable2
#     K_2 = GaussianMatrix(input2,sigma)/(input2.size(dim=0))
#     L_2,_ = torch.symeig(K_2,eigenvectors=True)
#     lambda_2 = torch.abs(L_2)
#     H_2 = (1/(1-alpha))*torch.log((torch.sum(lambda_2 ** alpha)))
    
    
#     input3 = variable3
#     K_3 = GaussianMatrix(input3,sigma)/(input3.size(dim=0))
#     L_3,_ = torch.symeig(K_3,eigenvectors=True)
#     lambda_3 = torch.abs(L_3)
#     H_3 = (1/(1-alpha))*torch.log((torch.sum(lambda_3 ** alpha)))
    
#     input4 = variable4
#     K_4 = GaussianMatrix(input4,sigma)/(input4.size(dim=0))
#     L_4,_ = torch.symeig(K_4,eigenvectors=True)
#     lambda_4 = torch.abs(L_4)
#     H_4 = (1/(1-alpha))*torch.log((torch.sum(lambda_4 ** alpha)))
    
#     input5 = variable5
#     K_5 = GaussianMatrix(input5,sigma)/(input5.size(dim=0))
#     L_5,_ = torch.symeig(K_5,eigenvectors=True)
#     lambda_5 = torch.abs(L_5)
#     H_5 = (1/(1-alpha))*torch.log((torch.sum(lambda_5 ** alpha)))
    
#     input6 = variable6
#     K_6 = GaussianMatrix(input6,sigma)/(input6.size(dim=0))
#     L_6,_ = torch.symeig(K_6,eigenvectors=True)
#     lambda_6 = torch.abs(L_6)
#     H_6 = (1/(1-alpha))*torch.log((torch.sum(lambda_6 ** alpha)))
    
    
#     input7 = variable7
#     K_7 = GaussianMatrix(input7,sigma)/(input7.size(dim=0))
#     L_7,_ = torch.symeig(K_7,eigenvectors=True)
#     lambda_7 = torch.abs(L_7)
#     H_7 = (1/(1-alpha))*torch.log((torch.sum(lambda_7 ** alpha)))
    
#     input8 = variable8
#     K_8 = GaussianMatrix(input8,sigma)/(input8.size(dim=0))
#     L_8,_ = torch.symeig(K_8,eigenvectors=True)
#     lambda_8 = torch.abs(L_8)
#     H_8 = (1/(1-alpha))*torch.log((torch.sum(lambda_8 ** alpha)))
    

    
#     K_xyz = K_1*K_2*K_3*K_4*K_5*K_6*K_7*K_8*(input1.size(dim=0))
#     K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
#     L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
#     lambda_xyz = torch.abs(L_xyz)
#     #lambda_xy = L_xy
#     H_xyz =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyz ** alpha)))
    
#     #mutual_information = H_x + H_y - H_xy
#     return H_1+H_2+H_3+H_4+H_5+H_6+H_7+H_8 - H_xyz

input_dim = 162
output_dim = 8
output_dim_2 = 5
h_n = 9000
w = 307
h=307
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(4, 8, kernel_size=6, stride=3)
#        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
#        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
#        #self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(input_dim, int(h_n/1))
        self.norm1 = nn.BatchNorm1d(h_n)
        self.fc2 = nn.Linear(int(h_n/1), int(h_n/1))
        self.norm2 = nn.BatchNorm1d(h_n)
        self.fc3 = nn.Linear(h_n, int(h_n/1))
        self.norm3 = nn.BatchNorm1d(h_n)
        self.fc4 = nn.Linear(int(h_n/1), int(h_n/1))
        self.norm4 = nn.BatchNorm1d(h_n)
        self.fc5 = nn.Linear(int(h_n/1), h_n)
        self.norm5 = nn.BatchNorm1d(h_n)
        self.fc6 = nn.Linear(h_n, int(h_n/1))
        self.norm6 = nn.BatchNorm1d(h_n)
        self.fc7 = nn.Linear(h_n, 128)
        self.norm7 = nn.BatchNorm1d(128)
        
        self.fc8 = nn.Linear(128, output_dim)
        self.fc9 = nn.Linear(128, output_dim_2)
        
    def single_power_step(self, A, x):
        x = torch.matmul(A, x)
        x = x/torch.norm(x)
        return x
    def alt_matrix_power(self, A, x, power):
        iter_count_tf = 0
        #condition  = lambda it, A, x: it< power
        #body = lambda it, A, x: (it+1, A, )
        #loop_vars = [iter_count_tf, A, x]
        it = 0
        while it<power:
            it+=1
            x = self.single_power_step(A, x)
        
        
        #output = tf.while_loop(condition, body, loop_vars)[2]
        e = torch.norm(torch.matmul(A, x))
        return x, e
    
    
    
    
    def alt_power_whitening(self, input_tensor, output_dim, n_iterations=250, **kwargs):
        R = torch.empty([output_dim,output_dim]).normal_(mean=0,std=1).cuda()
        W = torch.zeros([output_dim,output_dim]).cuda()
        input_tensor - input_tensor.mean(0)[None,:]
        C = torch.matmul(input_tensor.T, input_tensor)/input_tensor.shape[0]
        iter_count_tf = 0
        condition = lambda it, C, W, R: it<output_dim
        it = 0
        while it<output_dim:
            v, l = self.alt_matrix_power(C, R[:, it, None], n_iterations)
            it+=1
            C = C - l * torch.matmul(v, v.T)
            W = W + 1 / torch.sqrt(l) * torch.matmul(v, v.T)
        whitened_output = torch.matmul(input_tensor, W.T)
        return whitened_output, W, input_tensor.mean(0), C
        
        

    def forward(self, x):
        #x = self.alt_power_whitening(x, output_dim)[0]
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        x = F.relu(self.norm4(self.fc4(x)))
        x = F.relu(self.norm5(self.fc5(x)))
        x = F.relu(self.norm6(self.fc6(x)))
        x = F.relu(self.norm7(self.fc7(x)))
        
        #x = F.sigmoid(self.fc8(x))
        x1 = self.fc8(x)
        x1 = F.softmax(x1,-1)
        x1 = self.alt_power_whitening(x1, output_dim)[0]
        
        x2 = self.fc9(x)
        x2 = F.softmax(x2,-1)
        x2 = self.alt_power_whitening(x2, output_dim_2)[0]
        #x2 = F.softmax(x2,-1)
        return x1,x2
    
    
    
model = Net()
model = nn.DataParallel(model)
model.to(device)   


N =1000
crition = nn.MSELoss()
loss_list = []
current_loss_old =1000000

params = list(model.parameters())# + list(model2.parameters())
optimizer = optim.Adam(params, lr=0.0001)
k= 0 
show_id = 1
for ep in range(1):
    range_all = list(range(len(X_all)))
    permuted_all = list(np.random.permutation(range_all))
    X_all_shuffle = X_all[permuted_all]

    
    
    for it in range(int(len(X_all)/N)):
        #it = 10
        xinput = X_all_shuffle[int(it*N):int(it*N)+N].astype(np.float32)
        #xinput =X_all[np.random.randint(30):30000:30].astype(np.float32)# ##
        #xinput = X_all[np.random.choice(range(len(X)),(N,))].astype(np.float32)
        #xinput = xinput[permuted_bs]#[permuted_bs]
    
        optimizer.zero_grad()
        xinput = torch.tensor(xinput).cuda().to(device)
        latent_x, latent_x2= model(xinput) 
        sigma = 0.1#(N)**(-1/(4+(1)))
        alpha = 1.01#1.01
        

        #reco = model2(latent_x)
        

        
        #loss_rec = ((torch.abs(reco - xinput)).sum(-1)/reco.shape[-1]).sum()/reco.shape[0]
        if show_id%2 == 0: 
            loss = MMI8(latent_x[:,0].unsqueeze(-1), latent_x[:,1].unsqueeze(-1),latent_x[:,2].unsqueeze(-1),latent_x[:,3].unsqueeze(-1),latent_x[:,4].unsqueeze(-1)
                    ,latent_x[:,5].unsqueeze(-1),latent_x[:,6].unsqueeze(-1),latent_x[:,7].unsqueeze(-1),sigma,alpha)
        else:
            loss = MMI5(latent_x2[:,0].unsqueeze(-1), latent_x2[:,1].unsqueeze(-1),latent_x2[:,2].unsqueeze(-1),latent_x2[:,3].unsqueeze(-1),latent_x2[:,4].unsqueeze(-1)
                    ,sigma,alpha)
        #loss = crition(latent_x,truth)
        #loss_t = MI(truth[:,0].unsqueeze(-1), truth[:,1].unsqueeze(-1),sigma,sigma,alpha)
        loss.backward()
        optimizer.step()
        current_loss = loss.cpu().detach().numpy()
        loss_list.append(current_loss)
        k+=1
        if k%85 == 0:
            print(k)

            print([current_loss])
            #plt.figure(figsize=(8, 8))
            #plt.subplot(4, 1, 1)
            if show_id%2 == 0:
                show = np.zeros([(truth_data.T).shape[0],output_dim])
            else:
                show = np.zeros([(truth_data.T).shape[0],output_dim_2])
                
            for iij in range(int(len(X_all)/N)):
                
                X_sample = X_all_shuffle[iij:-1:int(int(w*h)/N)].astype(np.float32)#X.astype(np.float32)#
                #print(X_sample.shape)
                X_sample = torch.tensor(X_sample).cuda().to(device)
                if show_id%2 == 0:
                    show_si,_ = model(X_sample)
                    show_si = show_si.cpu().detach().numpy()
                else:
                    _,show_si = model(X_sample)#.detach().numpy()
                    show_si = show_si.cpu().detach().numpy()
                #show_si = F.softmax(show_si,-1).cpu().detach().numpy()
                # #show_si = sess.run(dense, feed_dict={encoding_x:X_sample,sample_number:N})
                # show_si[:,0]= (show_si[:,0]-show_si[:,0].min())/(show_si[:,0].max()-show_si[:,0].min())#(show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
                # show_si[:,1]= (show_si[:,1]-show_si[:,1].min())/(show_si[:,1].max()-show_si[:,1].min())#(show_si[:,1]-show_si[:,1].mean())/show_si[:,1].std()
                # show_si[:,2]= (show_si[:,2]-show_si[:,2].min())/(show_si[:,2].max()-show_si[:,2].min())
                # show_si[:,3]= (show_si[:,3]-show_si[:,3].min())/(show_si[:,3].max()-show_si[:,3].min())
                # show_si[:,4]= (show_si[:,4]-show_si[:,4].min())/(show_si[:,4].max()-show_si[:,4].min())
                #show_si[:,2]= (show_si[:,2]-show_si[:,2].mean())/show_si[:,2].std()
                # show_si[:,0]= (show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
                # show_si[:,1]= (show_si[:,1]-show_si[:,1].mean())/show_si[:,1].std()
                # show_si[:,2]= (show_si[:,2]-show_si[:,2].mean())/show_si[:,2].std()
                # show_si[:,3]= (show_si[:,3]-show_si[:,3].mean())/show_si[:,3].std()
                # show_si[:,4]= (show_si[:,4]-show_si[:,4].mean())/show_si[:,4].std()
                show[iij:-1:int(int(w*h)/N)] = show_si
            if show_id%2 == 0:
                show_real = np.zeros([(truth_data.T).shape[0],output_dim])
            else:
                show_real = np.zeros([(truth_data.T).shape[0],output_dim_2])
                
            for i in range(len(show)):
                show_real[permuted_all[i]] =  show[i]
            show = show_real
            
            # pl.subplot(311)
            # pl.plot(show_si[:,0])
            # pl.subplot(312)
            # pl.plot(show_si[:,1])
            # pl.subplot(313)
            # pl.plot(show_si[:,2])
            # pl.show()
            # plt.show()
    
            #encoding = sess.run(dense, feed_dict={encoding_x:X,sample_number:N})
            # for j in range(show.shape[1]):
            #     plt.plot(show[:, j], color = colors[j])
            show_list = []
            for j in range(len(show)):
                if 1:
                    show_list.append(show[j])
            show_si =np.array(show_list)     
            
            # show_si[:,0]= (show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
            # show_si[:,1]= (show_si[:,1]-show_si[:,1].mean())/show_si[:,1].std()
            # show_si[:,2]= (show_si[:,2]-show_si[:,2].mean())/show_si[:,2].std()
            # show_si[:,3]= (show_si[:,3]-show_si[:,3].mean())/show_si[:,3].std()
            # show_si[:,4]= (show_si[:,4]-show_si[:,4].mean())/show_si[:,4].std()
            
            
            if show_id%2 == 0:
                end_num = output_dim
            else:
                end_num = output_dim_2
            for endm in range(end_num):
#                #show_si[:,endm]= (show_si[:,endm]-show_si[:,endm].mean())/show_si[:,endm].std()
                 show_si[:,endm]= (show_si[:,endm]-show_si[:,endm].min())/(show_si[:,endm].max()-show_si[:,endm].min())#(show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
           
            
           # show_si[:,1]= (show_si[:,1]-show_si[:,1].min())/(show_si[:,1].max()-show_si[:,1].min())#(show_si[:,1]-show_si[:,1].mean())/show_si[:,1].std()
            # show_si[:,2]= (show_si[:,2]-show_si[:,2].min())/(show_si[:,2].max()-show_si[:,2].min())
            # show_si[:,3]= (show_si[:,3]-show_si[:,3].min())/(show_si[:,3].max()-show_si[:,3].min())
            # show_si[:,4]= (show_si[:,4]-show_si[:,4].min())/(show_si[:,4].max()-show_si[:,4].min())
            # show_si[:,5]= (show_si[:,5]-show_si[:,5].min())/(show_si[:,5].max()-show_si[:,5].min())
            # show_si[:,6]= (show_si[:,6]-show_si[:,6].min())/(show_si[:,6].max()-show_si[:,6].min())
            # show_si[:,7]= (show_si[:,7]-show_si[:,7].min())/(show_si[:,7].max()-show_si[:,7].min())
            show = show_si
            
            
            
                            #show_si = sess.run(dense, feed_dict={encoding_x:X_sample,sample_number:N})
#            show_si[:,0]= (show_si[:,0]-show_si[:,0].min())/(show_si[:,0].max()-show_si[:,0].min())#(show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
#            show_si[:,1]= (show_si[:,1]-show_si[:,1].min())/(show_si[:,1].max()-show_si[:,1].min())#(show_si[:,1]-show_si[:,1].mean())/show_si[:,1].std()
#            show_si[:,2]= (show_si[:,2]-show_si[:,2].min())/(show_si[:,2].max()-show_si[:,2].min())
#            show_si[:,3]= (show_si[:,3]-show_si[:,3].min())/(show_si[:,3].max()-show_si[:,3].min())
#            show_si[:,4]= (show_si[:,4]-show_si[:,4].min())/(show_si[:,4].max()-show_si[:,4].min())
            
            if show_id%2 == 0:

    
                #plt.imshow(np.concatenate((show[:,0].reshape(w,h),show[:,1].reshape(w,h),show[:,2].reshape(w,h),show[:,3].reshape(w,h),show[:,4].reshape(w,h)
                #                            ,show[:,5].reshape(w,h),show[:,6].reshape(w,h),show[:,7].reshape(w,h)),-1))
                #plt.show()
                
                
                asa = np.max(show,-1)
                aaa= show - np.repeat(np.expand_dims(asa,-1),8,-1)
                aaa[aaa ==0] =1
                aaa[aaa<=0] = 0
                show = aaa
                
                error_list = []
                
                index_list = [0,1,2,3,4,5,6,7]
                for itruth in range(5):
                    min_error = 10000000000000
                    for ishow in range(8):
                        single_truth = truth_img[:,itruth]
                        single_show = show[:,ishow]
                        
                        mse_error = np.sum(np.abs(single_truth - single_show))/len(single_show)
                        if min_error>mse_error:
                            min_error = mse_error
                            index_list[itruth] = ishow
                            #index_list[ishow] = itruth
                            
                            
                    error_list.append(min_error)
                print(error_list)
                print(np.mean(error_list))
                
                save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h),show[:,index_list[5]].reshape(w,h),show[:,index_list[6]].reshape(w,h),show[:,index_list[7]].reshape(w,h)),-1)
                save_img = (save_img*255).astype(np.uint8)
                cv2.imwrite('plots/'+str(k)+'.png',save_img)
                
                #cv2.imwrite('plots/'+str(k)+'.png',np.concatenate((show[:,0].reshape(w,h),show[:,1].reshape(w,h),show[:,2].reshape(w,h),show[:,3].reshape(w,h),show[:,4].reshape(w,h)
                #                            ,show[:,5].reshape(w,h),show[:,6].reshape(w,h),show[:,7].reshape(w,h)),-1))
                
            else:
                #plt.imshow(np.concatenate((truth_img[:,0].reshape(w,h),truth_img[:,1].reshape(w,h),truth_img[:,2].reshape(w,h),truth_img[:,3].reshape(w,h),truth_img[:,4].reshape(w,h)),-1))
                #plt.show()
                #plt.imshow(np.concatenate((show[:,0].reshape(w,h),show[:,1].reshape(w,h),show[:,2].reshape(w,h),show[:,3].reshape(w,h),show[:,4].reshape(w,h)
                #                            ),-1))
                #plt.show()
                
                asa = np.max(show,-1)
                aaa= show - np.repeat(np.expand_dims(asa,-1),5,-1)
                aaa[aaa ==0] =1
                aaa[aaa<=0] = 0
                show = aaa
                
                error_list = []
                
                index_list = [0,1,2,3,4]
                for itruth in range(5):
                    min_error = 10000000000000
                    for ishow in range(5):
                        single_truth = truth_img[:,itruth]
                        single_show = show[:,ishow]
                        
                        mse_error = np.sum(np.abs(single_truth - single_show))/len(single_show)
                        if min_error>mse_error:
                            min_error = mse_error
                            index_list[itruth] = ishow
                            #index_list[ishow] = itruth
                            
                            
                    error_list.append(min_error)
                print(error_list)
                print(np.mean(error_list))
                
                if 0:#os_i == 0:
                  index_list = [0,1,2,3,4]                
                
                save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h)),-1)
                
                #np.save('vectors.npy',show)
                save_show = np.zeros(show.shape)
                for idm in range(5):
                    save_show[:,idm] = show[:,index_list[idm]]
                  
                
                
                np.save('index_list.npy',index_list)
                np.save('current_results.npy',save_show)
                save_img = (save_img*255).astype(np.uint8)
                cv2.imwrite('plots/'+str(k)+'.png',save_img)
                
                
            #show_id+=1

    