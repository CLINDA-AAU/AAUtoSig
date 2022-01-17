# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import copy
from functions import plotsigs
from AAUtoSig_init import AAUtoSig, train_AAUtoSig

#because plots broke the kernel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mc = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\article_1\generated_data\DLBCL1001_trainset1_80p.csv', sep=',', index_col=0).transpose()

context = mc.columns
mutation = [s[2:5] for s in context]

x_train = mc.sample(frac=0.8)
x_test = mc.drop(x_train.index)



model = AAUtoSig(dim1 = 30, dim2 = 5)


# Validation using MSE Loss function
loss_function = torch.nn.MSELoss(reduction='mean')
#loss_function = torch.nn.KLDivLoss()
#Note the torch KL div assumes that the frist argument is already log-transformed


# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-3)#,
                             #weight_decay = 1e-8)
                             
train_AAUtoSig(epochs = 500, 
               model = model, 
               x_train = x_train, 
               x_test = x_test, 
               loss_function = loss_function, 
               optimizer = optimizer)
'''
W = best_model.dec1.weight.data    
W_array = W.numpy()


for i in range(n_sigs):
    plotsigs(context, mutation, W_array[:,i])    


validation_set = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\article_1\generated_data\DLBCL1001_testset1_20p.csv', sep=',', index_col=0).transpose()

x_validation_tensor = torch.tensor(validation_set.values, 
                             dtype = torch.float32)
res = best_model(x_validation_tensor)
print(loss_function(res,x_validation_tensor))
'''