'''This is my attempt to implement my interpretation of the model proposed in 'A Deep Non-negative Matrix Factorization Approach via
Autoencoder for Nonlinear Fault Detection' by Ren et al. 2020'''

from audioop import bias
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

class DNMF(torch.nn.Module):
    def __init__(self, nsig, batch_size, kernel_size = 3, stride = 1): #tri-gams
        super().__init__()
    	
        self.fc_dim = int(((96-kernel_size)/stride + 1 - kernel_size)/stride + 1)

        self.encoder = torch.nn.Sequential(
            # Building an encoder
            # 96 => dim1 => dim2
            torch.nn.Conv1d(batch_size, 3, kernel_size = kernel_size),
            torch.nn.BatchNorm1d(3, affine= False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(3, 6, kernel_size = kernel_size),
            torch.nn.BatchNorm1d(6, affine= False),
            torch.nn.ReLU(),
            torch.nn.Linear(6*self.fc_dim, 96),

        )
        
        #NMF part 
        # 272 => 15 => 272
        self.NMF1 = torch.nn.Linear(96, nsig, bias = False)

        self.NMF2 = torch.nn.Linear(nsig, 96, bias = False)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(96, 6*96),#if we begin at 6*96 we will end at fc_dim,
            #torch.nn.BatchNorm1d(6*self.fc_dim)  #this one is the issue
            # Building a decoder 
            # dim2 => dim1 => 96
            torch.nn.Conv1d(6, 3, kernel_size = kernel_size),
            torch.nn.BatchNorm1d(3, affine= False),
            torch.nn.ReLU(),
            torch.nn.Conv1d(3, batch_size, kernel_size = kernel_size),
            torch.nn.Linear(self.fc_dim, 96)
        )            

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1, 6 *self.fc_dim)
        x = self.NMF1(x)
        x = self.NMF2
        x = x.view(-1, 6, 96) #if we begin at 6*96 we will end at fc_dim
        x = self.decoder(x)
        return x
        
    # Model Initialization

net = DNMF(5, batch_size=2)

dat = torch.tensor([np.ones(96), np.ones(96)], dtype = torch.float)

asd = net(dat)


def train_DNMF(epochs, model, x_train, loss_function, optimizer, batch_size):
    
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    #x_test_tensor = torch.tensor(x_test.values, 
    #                             dtype = torch.float32)
    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    

    for _ in range(epochs):
        model.train()
        
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)#.view(-1,96))
            
          # Calculating the loss function
          loss = loss_function(reconstructed, data)#.view(-1,96))# + torch.mean(reconstructed) - torch.mean(data.view(-1,96))
          
          # The gradients are set to zero,
          # the the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        with torch.no_grad():

            for p in model.NMF1.weight + model.NMF2.weight: #These are the inverse signatures and signatures
                p.clamp_(min = 0)

    return(model)

data = 