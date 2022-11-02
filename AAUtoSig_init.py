# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy


class AAUtoSig(torch.nn.Module):
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
                                  
def train_AAUtoSig(epochs, model, x_train, x_test, loss_function, optimizer, batch_size, do_plot=False):
    
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    x_test_tensor = torch.tensor(x_test.values, 
                                 dtype = torch.float32)
    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    

    outputs = []
    training_plot=[]
    validation_plot=[]

    for epoch in range(epochs):
        train_loss = 0.0       
        model.train()
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)#.view(-1,96)
            
          # Calculating the loss function,
          loss = loss_function(reconstructed, data)#.view(-1,96)
          
          # The gradients are set to zero,
          # the the gradient is computed and stored.
          # .step() performs parameter update
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          train_loss += loss.item()
        training_plot.append(train_loss/len(trainloader))
        with torch.no_grad():
            for p in model.dec1.weight:
                p.clamp_(min = 0)
            model.eval()        
            inputs  = x_test_tensor
            outputs = model(inputs)
            valid_loss = loss_function(inputs, outputs)
        
            validation_plot.append(valid_loss)
            #print("Epoch {}, training loss {}, validation loss {}".format(epoch, 
            #                                                            training_plot[-1], 
            #                                                            validation_plot[-1]))
    if do_plot:
        plt.figure(figsize=(16,12))
        plt.subplot(3, 1, 1)
        plt.title('Score per epoch')
        #plt.ylabel('Kullback Leibler Divergence')
        plt.plot(list(range(len(training_plot))), validation_plot, label='Validation loss')
        plt.plot(list(range(len(training_plot))), training_plot, label='Train loss')
        plt.legend()
            
    
    return(model, validation_plot[-1].item(), training_plot[-1])
