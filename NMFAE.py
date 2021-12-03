# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 08:40:05 2021

@author: bjyw
"""


import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

import copy

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




#mc = pd.read_csv('Q:/AUH-HAEM-FORSK-MutSigDLBCL222/generated_data/WGS_PCAWG_96_LymphBNHL.csv', index_col=0)#.transpose()


mc = pd.read_csv('Q:/AUH-HAEM-FORSK-MutSigDLBCL222/generated_data/WGS_PCAWG_96_LymphBNHL.csv', index_col=0).transpose()

#mc = pd.read_csv('data/Ovarian_pooled.csv', index_col=0).transpose()


x_train = mc.sample(frac=0.8)
x_test = mc.drop(x_train.index)

x_train_tensor = torch.tensor(x_train.values, dtype = torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype = torch.float32)


#det går nok galt fordi det ikke er en tensor

#mc_tensor = torch.tensor(mc.values, dtype = torch.float32)
trainloader = torch.utils.data.DataLoader(x_train_tensor, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(x_test_tensor, batch_size=8)




# Creating linear (NMF autoencoder)
# 96 ==> 8 ==> 96
class NMFAE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        
        self.enc1 = torch.nn.Linear(96, dim, bias = False)
        self.dec1 = torch.nn.Linear(dim, 96, bias = False)
        
        # Building an linear encoder
        # 96 => dim
        #self.encoder = torch.nn.Sequential(
        #    torch.nn.Linear(96, dim, bias = False),
        #    torch.nn.Identity() #also a placeholder for cooler func
        #    )
          
        # Building an linear decoder 
        # dim ==> 96
        #self.decoder = torch.nn.Sequential(
        #    torch.nn.Linear(dim, 96, bias = False),
        #    torch.nn.Identity()
        #)
  
    #def forward(self, x):
    #    encoded = self.encoder(x)
    #    decoded = self.decoder(encoded)
    #    return decoded

    def forward(self, x):
        x = self.enc1(x)
        x = self.dec1(x)
        return x


# Model Initialization
model = NMFAE(dim = 7)
  
# Validation using MSE Loss function
#loss_function = torch.nn.MSELoss(reduction='mean')
loss_function = torch.nn.KLDivLoss()

  #
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-8)


#Train
epochs = 10000
outputs = []

training_plot=[]
validation_plot=[]

last_score=np.inf
max_es_rounds = 5
es_rounds = max_es_rounds
best_epoch= 0

for epoch in range(epochs):
    model.train()
    
    for data in trainloader:
      # Output of Autoencoder
      reconstructed = model(data.view(-1,96))
        
      # Calculating the loss function
      loss = loss_function(reconstructed, data.view(-1,96))
        
      # The gradients are set to zero,
      # the the gradient is computed and stored.
      # .step() performs parameter update
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      W = model.dec1.weight.data
     # print statistics
    with torch.no_grad():
        valid_loss=0
        train_loss=0
        model.eval()
        
        inputs = x_train_tensor[:]
        outputs = model(inputs)
        
        train_loss = loss_function(inputs, outputs)
        training_plot.append(train_loss)
    
 

        inputs  = x_test_tensor
        outputs = model(inputs)
        valid_loss = loss_function(inputs, outputs)
      
        validation_plot.append(valid_loss)
        print("Epoch {}, training loss {}, validation loss {}".format(epoch, 
                                                                      np.round(training_plot[-1],2), 
                                                                      np.round(validation_plot[-1],2)))

    

        
 #Patient early stopping - thanks to Elixir  
    if last_score > valid_loss:
        last_score = valid_loss
        best_epoch = epoch
        es_rounds = max_es_rounds
        best_model = copy.deepcopy(model)
    else:
        if es_rounds > 0:
            es_rounds -=1
        else:
            print('EARLY-STOPPING !')
            print('Best epoch found: nº {}'.format(best_epoch))
            print('Exiting. . .')
            break


plt.figure(figsize=(16,12))
plt.subplot(3, 1, 1)
plt.title('Score per epoch')
plt.ylabel('Kullback Leibler Divergence')
plt.plot(list(range(len(training_plot))), validation_plot, label='Validation DKL')
plt.plot(list(range(len(training_plot))), training_plot, label='Train DKL')
plt.legend()


for i in range(7):
    plt.figure(figsize=(16,12))
    plt.bar(mc.columns, W[:,i]/torch.sum(W[:,i]))


'''
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[-100:])
'''
