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


#from random import sample

from functions import  simulate_counts, cosine_HA, split_data
from AAUtoSig_init import AAUtoSig, train_AAUtoSig
import torch
from optuna_opt import optuna_tune
import os

print(" installed")

def out_errorNMF(train_df, validation_df, nsigs, true_sigs, criterion, epochs):
  if criterion == "KL" or criterion == "PoNNL":
    beta = 1
  if criterion == "MSE":
    beta = 2  
  model = NMF(n_components=nsigs, init='random', max_iter = epochs, solver = 'mu', beta_loss = beta)
  exposures = model.fit_transform(train_df)
  signatures = model.components_

  ref_exposures = model.transform(X = validation_df)
  rec = np.dot(ref_exposures, signatures)
  
  #if criterion == "KL" or criterion == "PoNNL":
  #    res = scipy.special.kl_div(validation_df, rec)
  #    out_error = res.mean().mean()
  if criterion == "MSE":
      out_error = np.mean(((validation_df - rec)**2).to_numpy())

  cos_NMF = cosine_HA(signatures, true_sigs.T)[0]
  cos_mean = np.mean(cos_NMF.diagonal())
  print(out_error)
  return cos_mean, out_error

def out_error_AAUtoSig(train_df, validation_df, nsigs, true_sigs, loss_name, optimizer_name, epochs):
    train = train_df.columns
    sigs = true_sigs.index 
    params = optuna_tune(train_df, nsigs, loss_name, optimizer_name)
    model = AAUtoSig(nsigs)

    if optimizer_name == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])
    if optimizer_name == "Tuned":
      optimizer =  getattr(torch.optim, params['optimizer'])(model.parameters(), lr = params['lr'])
   

    _, out_error, _ = train_AAUtoSig(epochs, model, train_df, validation_df, loss_name= loss_name, optimizer=optimizer, batch_size = params['batch_size'], do_plot = False, ES= False)

    signatures = model.dec1.weight.data    
    signatures = pd.DataFrame(signatures.numpy())

    if any(np.sum(signatures, axis = 0) == 0):
      #print(np.sum(signatures, axis = 0) )
      #print(signatures.loc[:, np.sum(signatures, axis = 0) == 0] )
      signatures.loc[:, np.sum(signatures, axis = 0) == 0] += 1e-5
      #print(np.sum(signatures, axis = 0) )

    cos_AE = cosine_HA(signatures.T, true_sigs.T)[0]
    cos_mean = np.mean(cos_AE.diagonal())
      
    return cos_mean, out_error

def performance_analysis(npatients, nsigs, loss_name, optimizer_name, epochs):
  mut_matrix_l, signatures_l, _ = simulate_counts(nsigs, npatients)
  print("generated_data")
  train_data_l, validation_data_l = split_data(mut_matrix_l.T, 0.8)
  train_data_l = train_data_l/train_data_l.max().max()
  validation_data_l = validation_data_l/train_data_l.max().max()
  cosineNMF_perm, outNMF = out_errorNMF(train_data_l, validation_data_l, nsigs, signatures_l, criterion = loss_name, epochs = epochs)
  print("NMF")
  cosineAE_perm, outAE = out_error_AAUtoSig(train_data_l, validation_data_l, nsigs, signatures_l, loss_name = loss_name, optimizer_name = optimizer_name, epochs = epochs)
  print("AE")
  return [cosineNMF_perm, outNMF, cosineAE_perm, outAE]



n_sims = 30
n_patients = 500
n_sigs = 7
epochs = 500
loss_name = "MSE"
optimizer_name = "Adam"
#data = (m.drop(['Unnamed: 0', '0'], axis = 1)).T


os.chdir(r"dfs_forskning/AUH-HAEM-FORSK-MutSigDLBCL222/article_1/scripts/AAUtoSig")
res = Parallel(n_jobs = 5)(delayed(performance_analysis)(n_patients, n_sigs, loss_name, optimizer_name, epochs) for i in range(n_sims))
print("analysis_done")
result = pd.DataFrame(res)
result.columns = ["NMF_perm", "outNMF", "AE_perm", "outAE" ]
print(result)
name = "Linear_"+ loss_name + "_ADAM_nsim:" + str(n_sims) + "_n_pat:" + str(n_patients) + "_nsigs:" + n_sigs + "_epochs:" + str(epochs)

matplotlib.use('Agg')
fig=plt.figure()
# set height of each subplot as 8
fig.set_figheight(15)
 
# set width of each subplot as 8
fig.set_figwidth(15)

spec = gridspec.GridSpec(ncols=2, nrows=2,
                         width_ratios=[2.3, 1], wspace=0.2,
                         hspace=0.2, height_ratios=[2.3, 1])

#fig.title('NMF vs AE performance on linearly simulated data')

ax1 = fig.add_subplot(spec[0])
plt.scatter(y = result['NMF_perm'], x = result['outNMF'], c = 'blue', label = 'NMF')
plt.scatter(y = result['AE_perm'], x = result['outAE'], c = 'red', label = 'AE')
#plt.xlabel('Out of sample error')
plt.ylabel('mean diagonal cosine')
plt.legend()
ax2 = fig.add_subplot(spec[1])
plt.boxplot(result[['NMF_perm', 'AE_perm']] ,labels = ["NMF", "AE"])
#plt.ylabel('mean diagonal cosine')
ax3 = fig.add_subplot(spec[2])
plt.boxplot(result[['outAE', 'outNMF']], labels = ["AE", "NMF"], vert=False)
plt.xlabel('Out of sample error')

ax1.get_shared_x_axes().join(ax1, ax3)
ax1.get_shared_y_axes().join(ax1, ax2)

plt.savefig(name + "scatter.png", transparent=True)


'''
matplotlib.use('Agg')
plt.boxplot(result[['outNMF', 'outAE']], labels = ["NMF", "AE"])
plt.title('NMF vs AE out of sample error on linearly simulated data')
plt.xlabel('Out of sample error')
plt.savefig(name + "_error.png", transparent=True)
print("plot_saved")
plt.clf()


plt.boxplot(result[['NMF_perm', 'AE_perm']] ,labels = ["NMF", "AE"])
plt.title('NMF vs AE cosine values on linearly simulated data')
plt.ylabel('mean diagonal cosine')
plt.savefig(name + "_cosine.png", transparent=True)
plt.clf()

plt.scatter(y = result['NMF_perm'], x = result['outNMF'], c = 'blue', label = 'NMF')
plt.scatter(y = result['AE_perm'], x = result['outAE'], c = 'red', label = 'AE')
plt.xlabel('Out of sample error')
plt.ylabel('mean diagonal cosine')
plt.title('NMF vs AE performance on linearly simulated data')
plt.legend()
plt.savefig(name + "scatter.png", transparent=True)

'''