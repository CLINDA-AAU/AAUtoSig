# -*- coding: utf-8 -*-

import torch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import NMF

import copy
from functions import plotsigs
from NMFAE_init import NMFAE, train_NMFAE

#because plots broke the kernel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mc = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\generated_data\Simulated_data\4sigs600pat14012022\mut_matrix.csv', sep=',',index_col=0).transpose()

context = mc.columns

mutation = [s[2:5] for s in context]
x_train = mc.sample(frac=0.8)
x_test = mc.drop(x_train.index)

n_sigs = 4

#-----------------------------------------------NMF--------------------------------------------------------------
nmf_model = NMF(n_components=n_sigs)
exposures = nmf_model.fit_transform(x_train)
signatures = nmf_model.components_
ref_exposures = nmf_model.transform(X = x_test)
rec = np.dot(ref_exposures, signatures)
nmf_MSE = np.mean(((x_test - rec)**2).to_numpy())
  
'''
for i in range(n_sigs):
  plotsigs(context, mutation, signatures[:,i])    

#-----------------------------------------------NMFAE------------------------------------------------------------
'''
model = NMFAE(dim1 = n_sigs)


# Validation using MSE Loss function
loss_function = torch.nn.MSELoss(reduction='mean')
#loss_function = torch.nn.KLDivLoss()
#Note the torch KL div assumes that the frist argument is already log-transformed


# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-3)#,
                             #weight_decay = 1e-8)
                             
train_NMFAE(epochs = 500, 
               model = model, 
               x_train = x_train, 
               loss_function = loss_function, 
               optimizer = optimizer)

W = model.dec1.weight.data    
W_array = W.numpy()


for i in range(n_sigs):
    plotsigs(context, mutation, W_array[:,i])    


validation_set = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\article_1\generated_data\DLBCL1001_testset1_20p.csv', sep=',', index_col=0).transpose()

x_validation_tensor = torch.tensor(validation_set.values, 
                             dtype = torch.float32)
res = model(x_validation_tensor)
print(loss_function(res,x_validation_tensor))
