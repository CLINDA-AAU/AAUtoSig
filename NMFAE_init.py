# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy


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
                                
def train_NMFAE(epochs, model, x_train, loss_function, optimizer):
    
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=16, 
                                              shuffle=True)
    
    outputs = []
    
    training_plot=[]    
    last_score=np.inf
    max_es_rounds = 50
    es_rounds = max_es_rounds
    best_epoch= 0
    #l1_lambda = 0.001
    
    for epoch in range(epochs):
        model.train()
        
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data.view(-1,96))
            
          # Calculating the loss function
          loss = loss_function(reconstructed, data.view(-1,96))# + torch.mean(reconstructed) - torch.mean(data.view(-1,96))

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        # print statistics
        with torch.no_grad():
            for p in model.parameters():
                p.clamp_(min = 0)
    
    return(model)