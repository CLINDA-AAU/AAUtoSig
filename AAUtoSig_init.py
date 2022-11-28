# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from datetime import date


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
        x = F.relu(self.dec1(x))
        return x
        
    # Model Initialization
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def kl_poisson(p, q): # p = inputs, q =  outputs?
    return torch.mean( torch.where(p != 0, p * torch.log(p / q) - p + q, 0))
                                  
def train_AAUtoSig(epochs, model, x_train, x_test, loss_name, optimizer, batch_size, do_plot=False, ES = True, i = None):
    if i is None:
        i = str(date.today())
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    x_test_tensor = torch.tensor(x_test.values, 
                                 dtype = torch.float32)
    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
   
    #if loss_name == "KL":
      #criterion = kl_poisson() #torch.nn.KLDivLoss(reduction = 'batchmean')
    if loss_name == "MSE":
       criterion = torch.nn.MSELoss()
    if loss_name == "PoNNL":
       criterion = torch.nn.PoissonNLLLoss()
    
    
    if ES:
        last_score=np.inf
        max_es_rounds = 50
        es_rounds = max_es_rounds
        best_epoch= 0 
    outputs = []
    training_plot=[]
    validation_plot=[]
    lr = get_lr(optimizer)
    for epoch in range(epochs):
        train_loss = 0.0       
        model.train()
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)#.view(-1,96)
            
          # Calculating the loss function,
          if loss_name == "KL":
            loss = kl_poisson(reconstructed, data)
          else:#if loss_name == "MSE":
            loss = criterion(reconstructed, data)#.view(-1,96)
          
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

            if loss_name == "KL":
               valid_loss = kl_poisson(inputs, outputs)
               #valid_loss = criterion(torch.log(outputs), inputs) + torch.mean(outputs) - torch.mean(inputs)
            else:#if loss_name == "MSE":
               valid_loss = criterion(outputs, inputs)#.view(-1,96)

        
            validation_plot.append(valid_loss)
        #Patient early stopping - thanks to Elixir  
        if ES:
            if last_score > valid_loss:
                last_score = valid_loss
                best_epoch = epoch
                es_rounds = max_es_rounds
                best_model = copy.deepcopy(model)
            
            else:
                if es_rounds > 0:
                    es_rounds -=1
                else:
                    print('EARLY STOPPING')
                    print('Best epoch found: nº {}'.format(best_epoch))
                    print('Exiting. . .')
                    break
    if do_plot:
        matplotlib.use('Agg')
        plt.figure(figsize=(16,12))
        plt.subplot(3, 1, 1)
        plt.title('LOSS:' + loss_name  + '_lr:' + str(np.round(lr, 5)))
        plt.plot(list(range(len(training_plot))), validation_plot, label='Validation loss')
        plt.plot(list(range(len(training_plot))), training_plot, label='Train loss')
        plt.legend()
        plt.savefig(i + "LOSS:" + loss_name + "_lr:" + str(np.round(lr, 5))+  "val_curve.png", transparent=True)
        plt.clf()
        
    if not ES:
        best_model = model  
        last_score = validation_plot[-1]  
    return(best_model, last_score.item(), training_plot[-1])
