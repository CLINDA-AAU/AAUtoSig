# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np          

class NMFAE(torch.nn.Module):
    def __init__(self, dim1):
    
        super().__init__()

        # Building an linear encoder
        # 96 => dim1 => dim2
        self.enc1 = torch.nn.Linear(96, dim1, bias = False)
          
        # Building an linear decoder 
        # dim ==> 96
        self.dec1 = torch.nn.Linear(dim1, 96, bias = False)
            

    def forward(self, x):
        x = self.enc1(x)
        x = self.dec1(x)
        return x
        
    # Model Initialization
                                
def train_NMFAE(epochs, model, x_train, loss_function, optimizer, batch_size):
    
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    
    outputs = []

    training_plot = []
    validation_plot = []
    
    last_score = np.inf
    max_es_rounds = 50
    es_rounds = max_es_rounds
    best_epoch= 0    

    for epoch in range(epochs):
        model.train()
        
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)#.view(-1,96))
            
          # Calculating the loss function
          loss = loss_function(reconstructed, data)#.view(-1,96))

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        with torch.no_grad():
            for p in model.parameters():
                p.clamp_(min = 0)

            model.eval()        
            inputs = x_train_tensor[:]
            outputs = model(inputs)
            
            train_loss = loss_function(outputs,inputs) #+ torch.mean(reconstructed) - torch.mean(data.view(-1,96))
            #train_loss = kl_poisson(inputs, outputs)
    
            training_plot.append(train_loss)

    #plt.plot(list(range(len(training_plot))), training_plot, label='Train MSE')
    #plt.legend()
    #plt.show()
    #plt.show() 
    return(model)

'''
nsigs = 5

mf_df, true_sigs,_ = simulate_counts(nsigs, 5000, pentanucelotide = False)
trinucleotide = mf_df.index
mutation = [t[2:5] for t in trinucleotide]

X = mf_df.T

#80/20 train/validation split
x_train = X.sample(frac=0.8)
x_val = X.drop(x_train.index)

#choosing the 'true' number of signatures
model = NMFAE(dim1 = nsigs)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss(reduction='mean')

# Using an Adam Optimizer with lr = 1e-3
optimizer_enc = EGPM(model.dec1.parameters(), lr = 0.0001 , u_scaling=100, 
                       norm_per=None, gradient_clipping=True, 
                       weight_regularization=None, plus_minus=False,
                       init='bootstrap')

                            
train_NMFAE(epochs = 2000, 
            model = model, 
            x_train = x_train, 
            loss_function = loss_function, 
            optimizer = optimizer_enc,
            batch_size = 32)

'''
