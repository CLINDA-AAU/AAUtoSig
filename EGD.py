
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from functions import simulate_counts, plotsigs

class EGD_init(torch.nn.Module):
    def __init__(self, dim1):
    
        super().__init__()

        # Building an linear encoder
        # 96 => dim1
        self.enc1 = torch.nn.Linear(96, dim1, bias = False)
          
        # Building an linear decoder 
        # dim1 ==> 96
        self.dec1 = torch.nn.Linear(dim1, 96, bias = False)

    def forward(self, x):
        x = torch.nn.functional.relu(self.enc1(x))
        x = self.dec1(x)
        return x

    # Model Initialization
                                
def train_EGD(epochs, model, x_train, loss_function, optimizer_enc, 
                optimizer_dec, batch_size):
    
    #turn the training data into a tensor
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    
    #this is what loads makes the updates batch-wise insted of the full data matrix
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    
    loss_list = []
    for epoch in range(epochs):
        model.train() #set model in traning mode (alternative model.eval())
        loss_p = 0
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)
            
          # Calculating the loss function
          loss = loss_function(reconstructed, data)
          loss_p =+ loss.item()

          optimizer_enc.zero_grad() #clear old gradients
          optimizer_dec.zero_grad()
          
          loss.backward() #backpropagation
          
          optimizer_enc.step()#update params
          optimizer_dec.step()

        loss_list.append(loss_p)

    plt.plot(range(epochs), loss_list)
    plt.show() 
    return(model)