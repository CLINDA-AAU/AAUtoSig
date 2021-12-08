import torch
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import copy

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
# 96 ==> 8 ==> 96
class NMFAE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        
        # Building an linear encoder
        # 96 => dim
        self.enc1 = torch.nn.Sequential(
            torch.nn.Linear(96, dim, bias = False),
            torch.nn.Identity() #also a placeholder for cooler func
            )
          
        # Building an linear decoder 
        # dim ==> 96
        self.dec1 = torch.nn.Sequential(
            torch.nn.Linear(dim, 96, bias = False),
            torch.nn.Identity()
        )
  
    #def forward(self, x):
    #    encoded = self.encoder(x)
    #    decoded = self.decoder(encoded)
    #    return decoded

    def forward(self, x):
        x = self.enc1(x)
        x = self.dec1(x)
        return x