import torch
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import copy
from plots import plotsigs

#because plots broke the kernel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mc = pd.read_csv('Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\DLBCL_1001\DLBCL_mut_matrix.tsv', sep='\t', index_col=0).transpose()

context = mc.columns
mutation = [s[2:5] for s in context]

x_train = mc.sample(frac=0.8)
x_test = mc.drop(x_train.index)

x_train_tensor = torch.tensor(x_train.values, dtype = torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype = torch.float32)

trainloader = torch.utils.data.DataLoader(x_train_tensor, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(x_test_tensor, batch_size=8)


# Creating linear (NMF autoencoder)
# 96 ==> dim ==> 96
class NMFAE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.enc1 = torch.nn.Linear(96, dim, bias = False)
        self.dec1 = torch.nn.Linear(dim, 96, bias = False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.dec1(x)
        return x

# Model Initialization
n_sigs = 5
model = NMFAE(dim = n_sigs)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss(reduction='mean')
#loss_function = torch.nn.KLDivLoss()


# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-8)


#Train
epochs = 500
outputs = []

training_plot=[]
validation_plot=[]

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
      loss = loss_function(reconstructed, data.view(-1,96))
      
      # l1_norm = sum(p.abs().sum()
      #            for p in model.parameters())
 
      # loss = loss + l1_lambda * l1_norm
      
      # The gradients are set to zero,
      # the the gradient is computed and stored.
      # .step() performs parameter update
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    # print statistics
    with torch.no_grad():
        for p in model.parameters():
            p.clamp_(min = 0)
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


W = best_model.dec1.weight.data    
W_array = W.numpy()

for i in range(n_sigs):
    plotsigs(context, mutation, W_array[:,i])

H  = np.matmul(x_train, W_array).to_numpy()


