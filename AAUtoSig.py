# -*- coding: utf-8 -*-

from black import out
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import copy
from functions import plotsigs, simulate_counts
#from AAUtoSig_init import AAUtoSig, train_AAUtoSig
from NMFAE_init import NMFAE, train_NMFAE

from sklearn.model_selection import KFold

mc,_,_ = simulate_counts(5, 1000)

mc = mc.transpose()
context = mc.columns
mutation = [s[2:5] for s in context]

x_train = mc.sample(frac=0.8)
x_test = mc.drop(x_train.index)

lr = [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
nsigs = list(range(2,11))

kf = KFold(n_splits=5)

cv_res = []
for n in nsigs:
    for r in lr:
        out_err = []
        for train, test in kf.split(x_train):
            model = NMFAE(dim1=n)

            loss_function = torch.nn.MSELoss(reduction='mean')

            optimizer = torch.optim.Adam(model.parameters(), lr = r)

            train_NMFAE(epochs = 200, 
               model = model, 
               x_train = pd.DataFrame(x_train).iloc[train,:], 
               loss_function = loss_function, 
               optimizer = optimizer,
               batch_size = 16)

            cv_test_tensor = torch.tensor(pd.DataFrame(x_train).iloc[test,:].values, 
                                            dtype = torch.float32)

            cv_fit = model(cv_test_tensor)
            out_err.append(float(loss_function(cv_fit,cv_test_tensor).detach().numpy()))

        cv_res.append([n, r, np.mean(out_err)])
            

print(cv_res)




'''
#initializing cv is an int gives stratified Kfold cv
gs = GridSearchCV(net, params, refit=False, scoring='accuracy', verbose=1, cv=10)

gs.fit(x_train, x_train)

n_sigs = 5

model = AAUtoSig(dim1 = 30, dim2 = n_sigs)


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