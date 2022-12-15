#from click import option
#from scipy.optimize import nnls
#from functools import partial

import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
#import scipy 
from joblib import Parallel, delayed
from datetime import date


#from random import sample

from functions import  simulate_counts, cosine_HA, split_data
from AAUtoSig_init import AAUtoSig
import torch
from optuna_opt import optuna_tune
import os


def train_AAUtoSig(epochs, model, x_train, criterion, optimizer, batch_size, non_negative):
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)

    
    trainloader = torch.utils.data.DataLoader(x_train_tensor, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    
    

    for epoch in range(epochs):
        train_loss = 0.0       
        model.train()
        for data in trainloader:
          # Output of Autoencoder
          reconstructed = model(data)#.view(-1,96)
          loss = criterion(reconstructed, data)#.view(-1,96)
          
          # The gradients are set to zero,
          # the the gradient is computed and stored.
          # .step() performs parameter update
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          train_loss += loss.item()
        train_loss_total = train_loss/len(trainloader)
        with torch.no_grad():
            if non_negative == "all":
                for p in model.parameters():#model.dec1.weight:
                    p.clamp_(min = 0)
            if non_negative == "bases":
                for p in model.dec1.weight:
                    p.clamp_(min = 0)
            model.eval()        
    return(model, train_loss_total)


def in_errorNMF(train_df, nsigs, true_sigs, criterion, epochs):
  if criterion == "KL" or criterion == "PoNNL":
    beta = 1
  if criterion == "MSE":
    beta = 2  
  model = NMF(n_components=nsigs, init='random', max_iter = epochs, solver = 'mu', beta_loss = beta)
  exposures = model.fit_transform(train_df)
  signatures = model.components_
  in_error = model.reconstruction_err_

  cos_NMF = cosine_HA(signatures, true_sigs.T)[0]
  cos_mean = np.mean(cos_NMF.diagonal())
  return cos_mean, in_error

def in_error_AAUtoSig(train_df, nsigs, true_sigs, loss_name, optimizer_name, epochs, non_negative):
    #bemærk her at du tuner til at performe godt out of sample 
    params = optuna_tune(train_df, nsigs, loss_name, optimizer_name)
    model = AAUtoSig(96, nsigs, non_negativity = "ReLU")

    if optimizer_name == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])
    if optimizer_name == "Tuned":
      optimizer =  getattr(torch.optim, params['optimizer'])(model.parameters(), lr = params['lr'])
   

    _, in_error = train_AAUtoSig(epochs, model, train_df, loss_name= loss_name, optimizer=optimizer, batch_size = params['batch_size'], non_negative = non_negative)

    signatures = model.dec1.weight.data    
    signatures = pd.DataFrame(signatures.numpy())

    cos_AE = cosine_HA(signatures.T, true_sigs.T)[0]
    cos_mean = np.mean(cos_AE.diagonal())
      
    return cos_mean, in_error

def performance_analysis(npatients, nsigs, loss_name, optimizer_name, epochs, non_negative):
  mut_matrix_l, signatures_l, _ = simulate_counts(nsigs, npatients)
  train_data_l = mut_matrix_l.T/mut_matrix_l.T.max().max()
  cosineNMF_perm, outNMF = in_errorNMF(train_data_l, nsigs, signatures_l, criterion = loss_name, epochs = epochs)
  print("NMF")
  cosineAE_perm, outAE = in_error_AAUtoSig(train_data_l, nsigs, signatures_l, loss_name = loss_name, optimizer_name = optimizer_name, epochs = epochs, non_negative = non_negative)
  print("AE")
  return [cosineNMF_perm, outNMF, cosineAE_perm, outAE]



n_sims = 50
n_patients = 5000
n_sigs = 7
epochs = 5000
loss_name = "MSE"
optimizer_name = "Adam"
#data = (m.drop(['Unnamed: 0', '0'], axis = 1)).T


os.chdir(r"dfs_forskning/AUH-HAEM-FORSK-MutSigDLBCL222/article_1/scripts/AAUtoSig")
res = Parallel(n_jobs = 10)(delayed(performance_analysis)(n_patients, n_sigs, loss_name, optimizer_name, epochs, None) for i in range(n_sims))
print("analysis_done")
result = pd.DataFrame(res)
result.columns = ["NMF_perm", "inNMF", "AE_perm", "inAE" ]
print(result)
name = "Linear_"+ loss_name + "_ADAM_nsim:" + str(n_sims) + "_n_pat:" + str(n_patients) + "_nsigs:" + str(n_sigs) + "_epochs:" + str(epochs) + " NN:" + "none"

matplotlib.use('Agg')
fig=plt.figure()
# set height of each subplot as 15
fig.set_figheight(15)
 
# set width of each subplot as 15
fig.set_figwidth(15)

spec = gridspec.GridSpec(ncols=2, nrows=2,
                         width_ratios=[2.3, 1], wspace=0.2,
                         hspace=0.2, height_ratios=[2.3, 1])

#fig.title('NMF vs AE performance on linearly simulated data')

ax1 = fig.add_subplot(spec[0])
plt.scatter(y = result['NMF_perm'], x = result['inNMF'], c = 'blue', label = 'NMF')
plt.scatter(y = result['AE_perm'], x = result['inAE'], c = 'red', label = 'AE')
#plt.xlabel('Out of sample error')
plt.ylabel('mean diagonal cosine')
plt.legend()
ax2 = fig.add_subplot(spec[1])
plt.boxplot(result[['NMF_perm', 'AE_perm']] ,labels = ["NMF", "AE"])
#plt.ylabel('mean diagonal cosine')
ax3 = fig.add_subplot(spec[2])
plt.boxplot(result[['inAE', 'inNMF']], labels = ["AE", "NMF"], vert=False)
plt.xlabel('reconstruction error')

ax1.get_shared_x_axes().join(ax1, ax3)
ax1.get_shared_y_axes().join(ax1, ax2)
cwd = os.getcwd()
print(cwd)
plt.savefig(name + "scatter.png", transparent=True)
