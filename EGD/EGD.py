
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

class EGD_init(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim):
    
        super().__init__()

        # Building an linear encoder
        # 96 => dim1
        self.enc1 = torch.nn.Linear(input_dim, hidden_dim, bias = False)
          
        # Building an linear decoder 
        # dim1 ==> 96
        self.dec1 = torch.nn.Linear(hidden_dim, input_dim, bias = False)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.dec1(x)
        return x

    # Model Initialization
                                
def train_EGD(epochs, model, x_train, loss_function, optimizer, batch_size): #optimizer_enc, optimizer_dec , ):
    
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
        loss_p = 0.0
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)
            
          # Calculating the loss function

          #optimizer_enc.zero_grad() #clear old gradients
          #optimizer_dec.zero_grad()
          optimizer.zero_grad()


          loss = loss_function(reconstructed, data)
          loss_p =+ loss.item()
          
          loss.backward() #backpropagation
          
          #optimizer_enc.step()#update params
          #optimizer_dec.step()
          optimizer.step()

        loss_list.append(loss_p/x_train_tensor.shape[0])

    plt.plot(range(epochs), loss_list)
    plt.show() 
    return(model)