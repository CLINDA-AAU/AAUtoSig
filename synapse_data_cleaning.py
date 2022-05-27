# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from functions import cosine_perm, plotsigs

COSMIC = pd.read_csv(r'COSMIC\COSMIC_v3.2_SBS_GRCh37.txt', sep = '\t', index_col=0)
context = COSMIC.index
mutation = [s[2:5] for s in context]
COSMIC['mutation'] = mutation
COSMIC = COSMIC.sort_values('mutation')
mutation = COSMIC['mutation']
context = COSMIC.index
COSMIC = COSMIC.drop('mutation', axis = 1)

PCAWG = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\Alexandrov_2020_synapse\WGS_PCAWG_2018_02_09\WGS_PCAWG.96.csv')
PCAWG.index = [t[0] + '[' + m + ']' + t[2] for (t,m) in zip(PCAWG['Trinucleotide'], PCAWG['Mutation type'])]
PCAWG = PCAWG.drop(['Trinucleotide', 'Mutation type'], axis = 1)

cancers = [c.split('::')[0] for c in PCAWG.columns]
PatientID = [c.split('::')[1] for c in PCAWG.columns]

encoder = OneHotEncoder(sparse=(False))
cancer_one_hot_df_PCAWG = pd.DataFrame(encoder.fit_transform(np.array(cancers).reshape(-1, 1)))
cancer_one_hot_df_PCAWG.columns = encoder.get_feature_names()
cancer_one_hot_df_PCAWG.index = PatientID


TCGA = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\Alexandrov_2020_synapse\WES_TCGA_2018_03_09\WES_TCGA.96.csv')
TCGA.index = [t[0] + '[' + m + ']' + t[2] for (t,m) in zip(TCGA['Trinucleotide'], TCGA['Mutation type'])]
TCGA = TCGA.drop(['Trinucleotide', 'Mutation type'], axis = 1)

cancers = [c.split('::')[0] for c in TCGA.columns]
PatientID = [c.split('::')[1] for c in TCGA.columns]

cancer_one_hot_df_TCGA = pd.DataFrame(encoder.fit_transform(np.array(cancers).reshape(-1, 1)))
cancer_one_hot_df_TCGA.columns = encoder.get_feature_names()
cancer_one_hot_df_TCGA.index = PatientID

PCAWG = PCAWG.T
PCAWG['WGS'] = 1

TCGA = TCGA.T
TCGA['WGS'] = 0 

total = pd.concat([PCAWG, TCGA])


class AAUtoSig(torch.nn.Module):
    def __init__(self, dim1):
        super().__init__()

        
        # Building an linear encoder
        # 96 => dim1 => dim2
        self.enc1 = torch.nn.Linear(97, dim1, bias = False)
          
        # Building an linear decoder 
        # dim2 => dim1 => 96
        self.dec2 = torch.nn.Linear(dim1, 97, bias = False)
            

    def forward(self, x):
        x = self.enc1(x)
        x = self.dec2(x)
        return x
        
    # Model Initialization
                                
def train_AAUtoSig(epochs, model, x_train, loss_function, optimizer, batch_size):
    
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)

 
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
            for p in model.dec2.weight:
                p.clamp_(min = 0)
          
    return(model)
'''
def plotsigs(context, mutation, signatures, nsigs, title):
    colors = {'C>A': 'r', 'C>G': 'b', 'C>T': 'g', 
                'T>A' : 'y', 'T>C': 'c','T>G' : 'm' , 'WGS': 'k'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    for i in range(nsigs): signatures[:,i] = signatures[:,i]/np.sum(signatures[:,i])
    max_val = signatures.max().max()
    for i in range(nsigs):
        plt.subplot(nsigs,1, (i+1))
        #plt.figure(figsize=(20,7))
        plt.bar(x = context, 
                height =  signatures[:,i], 
                color = [colors[i] for i in mutation])
        plt.xticks([])
        plt.ylim( [ 0, max_val ] ) 
        if i == 0:
            plt.title(title)
    #plt.legend(handles,labels)
    plt.xticks(rotation=90)
    plt.show()
'''    
    
model = AAUtoSig(78)    
    
loss_function = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-3)#,
                             #weight_decay = 1e-8)

train_AAUtoSig(epochs = 500, model = model, x_train = total, loss_function = loss_function, 
                optimizer = optimizer, batch_size = 16)

sigs_est = model.dec2.weight.data
sigs_est = pd.DataFrame(sigs_est.numpy())#.iloc[:,0:5]

trinucleotide = total.columns
mutation =  [s[2:5] for s in trinucleotide[0:96]]
mutation.append('WGS')

sigs_est.index = trinucleotide

plt.hist(sigs_est.iloc[96,:])

#perm = cosine_perm(sigs_est.drop(['WGS']).T, COSMIC.T)

#sigs_est = sigs_est.iloc[0:96, perm[1]]
'''

context = COSMIC.index
mutation = [s[2:5] for s in context]

plotsigs(context, mutation, sigs_est.to_numpy(), 5, "Estimated signatures")  
plotsigs(context, mutation, COSMIC.to_numpy(), 5, "COSMIC signatures")  

signatures = COSMIC.columns
res = perm[0]


fig, ax = plt.subplots()
im = ax.imshow(res)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(signatures)))
ax.set_xticklabels(signatures)
ax.set_yticks(np.arange(len(signatures)))
ax.set_yticklabels(signatures)


# Loop over data dimensions and create text annotations.
for i in range(len(signatures)):
    for j in range(len(signatures)):
        text = ax.text(j, i, res[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()
'''
