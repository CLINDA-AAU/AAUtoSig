'''This is my attempt to implement my interpretation of the model proposed in 'A Deep Non-negative Matrix Factorization Approach via
Autoencoder for Nonlinear Fault Detection' by Ren et al. 2020'''

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import simulate_mixedLittle, plotsigs

class DNMF(torch.nn.Module):
    def __init__(self, nsig): 
        super().__init__()
    	
        self.encoder = torch.nn.Sequential(
            # Building an encoder
            torch.nn.Linear(96, 75),
            torch.nn.ELU(),
            torch.nn.Linear(75, 50),
            torch.nn.ELU(),
            torch.nn.Linear(50, 25),
            torch.nn.ELU(),
            torch.nn.Linear(25, 10),
            torch.nn.ELU()

        )
        self.expand_to_NMF = torch.nn.Linear(10, 96)
        #NMF part 
        # 272 => 15 => 272
        self.NMF1 = torch.nn.Linear(96, nsig, bias = False)

        self.NMF2 = torch.nn.Linear(nsig, 96, bias = False)

        self.decrease_to_conv = torch.nn.Linear(96, 10)
        self.decoder = torch.nn.Sequential(
            # Building a decoder 
            torch.nn.Linear(10, 25),
            torch.nn.ELU(),
            torch.nn.Linear(25, 50),
            torch.nn.ELU(),
            torch.nn.Linear(50, 75),
            torch.nn.ELU(),
            torch.nn.Linear(75, 96),
            torch.nn.ELU()
        )            

    def forward(self, x):
        x = self.encoder(x)
        x = self.expand_to_NMF(x)
        x = self.NMF1(x)
        x = self.NMF2(x)
        x = self.decrease_to_conv(x)
        x = self.decoder(x)
        return x
        
    # Model Initialization

#net = DNMF(5, batch_size=2)

#dat = torch.tensor([np.ones(96), np.ones(96)], dtype = torch.float)

#asd = net(dat)


def train_DNMF(epochs, model, x_train, loss_function, optimizer, batch_size):
    
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)
    #x_test_tensor = torch.tensor(x_test.values, 
    #                             dtype = torch.float32)
    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    
    loss_list = []
    for epoch in range(epochs):

        if int(round(1000*epoch/epochs,0)) % 100 == 0:
            print(str(round(100*epoch/epochs,0)) + "%" )
    
        model.train()
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data).view(-1,96)
                
          # Calculating the loss function
          loss = loss_function(reconstructed, data)#.view(-1,96))# + torch.mean(reconstructed) - torch.mean(data.view(-1,96))

          
          # The gradients are set to zero,
          # the the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        with torch.no_grad():
            for p in model.NMF2.weight: #These are the inverse signatures and signatures
                p.clamp_(min = 0) #fix this pls
    
    return model

data, sigs = simulate_mixedLittle(5, 600)
data = data.transpose()

#I think this model depends on the number of observations being divisible by batch size
model = DNMF(5)

loss_function = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-3)#,
                             #weight_decay = 1e-8)

train_DNMF(epochs = 500, model = model, x_train = data, loss_function = loss_function, optimizer = optimizer, batch_size = 16)
#print(model.NMF2.weight)

sigs_est = model.NMF2.weight.data
sigs_est = pd.DataFrame(sigs_est.numpy())
trinucleotide = sigs.index
mutation =  [s[2:5] for s in trinucleotide]


plotsigs(trinucleotide, mutation, sigs.to_numpy(), 5, "True signatures")  
plotsigs(trinucleotide, mutation, sigs_est.to_numpy(), 5, "Estimated signatures")  
