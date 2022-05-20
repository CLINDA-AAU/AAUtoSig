# -*- coding: utf-8 -*-

from black import out
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from AAUtoSig_init import train_AAUtoSig
from functions import cosine_perm, plotsigs, simulate_counts


mc, sigs,_ = simulate_counts(5, 1000)

mc = mc.transpose()
context = mc.columns
mutation = [s[2:5] for s in context]

x_train = mc.sample(frac=0.8)
x_test = mc.drop(x_train.index)

class AAUtoSig(torch.nn.Module):
    def __init__(self, dim1):
        super().__init__()

        
        # Building an linear encoder
        # 96 => dim1 => dim2
        self.enc1 = torch.nn.Linear(96, 200, bias = False)
        self.enc2 = torch.nn.Linear(200, 150, bias = False)
        self.enc3 = torch.nn.Linear(150, 100, bias = False)
        self.enc4 = torch.nn.Linear(100, 50, bias = False)
        self.enc5  = torch.nn.Linear(50, dim1, bias = False)


          
        # Building an linear decoder 
        # dim2 => dim1 => 96
        self.dec2 = torch.nn.Linear(dim1, 96, bias = False)
            

    def forward(self, x):
        x = F.softplus(self.enc1(x))
        x = F.softplus(self.enc2(x))
        x = F.softplus(self.enc3(x))
        x = F.softplus(self.enc4(x))
        x = F.softplus(self.enc5(x))
        x = F.softplus(self.dec2(x))
        return x


def plotsigs(context, mutation, signatures, nsigs, title):
    colors = {'C>A': 'r', 'C>G': 'b', 'C>T': 'g', 
                'T>A' : 'y', 'T>C': 'c','T>G' : 'm' , 'WGS': 'k'}
    context = context.append('WGS')
    mutation = mutation.append('WGS')
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    for i in range(nsigs): signatures[:,i] = signatures[:,i]/np.sum(signatures[:,i])
    max_val = signatures.max().max()
    for i in range(nsigs):
        plt.subplot(nsigs,1, (i+1))
        #plt.figure(figsize=(20,7))
        plt.bar(x = context, 
                height =  signatures[:,i], 
                color = [colors[i] for i in mutation])
        plt.xticks([])
        plt.ylim( [ 0, max_val ] ) 
        if i == 0:
            plt.title(title)
    #plt.legend(handles,labels)
    #plt.xticks(rotation=90)
    plt.show()
    


        
# Model Initialization
        

model = AAUtoSig(70)

loss_function = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-3)#,
                             #weight_decay = 1e-8)

train_AAUtoSig(epochs = 500, model = model, x_train = mc, loss_function = loss_function, 
                optimizer = optimizer, batch_size = 16)

sigs_est = model.dec2.weight.data
sigs_est = pd.DataFrame(sigs_est.numpy())
trinucleotide = sigs.index
mutation =  [s[2:5] for s in trinucleotide]

plotsigs(trinucleotide, mutation, sigs_est.to_numpy(), 5, "Estimated signatures")  



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