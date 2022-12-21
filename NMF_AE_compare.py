import os
import sys

import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import scipy 

import numpy as np
import ast
import torch
from sklearn.decomposition import NMF

from random import sample

from functions import cosine_HA
from AAUtoSig_init import AAUtoSig, train_AAUtoSig
from optuna_opt import optuna_tune

from joblib import Parallel, delayed
import multiprocessing
from datetime import date

today = date.today()

PCAWG = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\Alexandrov_2020_synapse\WGS_PCAWG_2018_02_09\WGS_PCAWG.96.csv')
PCAWG.index = [t[0] + '[' + m + ']' + t[2] for (t,m) in zip(PCAWG['Trinucleotide'], PCAWG['Mutation type'])]
PCAWG = PCAWG.drop(['Trinucleotide', 'Mutation type'], axis = 1)

cancers = [c.split('::')[0] for c in PCAWG.columns]
idx = [c == 'Prost-AdenoCA' for c in cancers]
PCAWG_prost = PCAWG.iloc[:, idx]

def split_data(data, frac_1):
  x_train = data.sample(frac = frac_1)
  x_validation = data.drop(x_train.index)
  return x_train, x_validation
  

def compare_NMF_AE(train_df, validation_df, nsigs, epochs):
   #define models
   model_NMF = NMF(n_components=nsigs, init='random', max_iter = epochs, solver='mu')
   
   model_AE = AAUtoSig(96, nsigs, non_negativity = None)
   loss_AE = torch.nn.MSELoss()
   params_AE = optuna_tune(train_df, nsigs, loss_AE)
   optimizer_AE = torch.optim.Adam(model_AE.parameters(), lr = params_AE['lr'])
   
   # fit and extract factor matrices
   train_AAUtoSig(epochs, model_AE, train_df, validation_df, criterion = loss_AE, optimizer=optimizer_AE, batch_size = params_AE['batch_size'], do_plot=False)

   exposures_NMF = model_NMF.fit_transform(train_df)
   signatures_NMF = model_NMF.components_

   train_tensor = torch.tensor(train_df.values,
                                    dtype = torch.float32)
   signatures_AE = model_AE.dec1.weight.data    
   signatures_AE = pd.DataFrame(signatures_AE.numpy())
   exposures_AE = model_AE.enc1.weight.data@train_df.T

   
   # calculate out-of-sample error
   ref_exposures_NMF = model_NMF.transform(X = validation_df)
   rec_NMF = np.dot(ref_exposures_NMF, signatures_NMF)
   out_error_NMF = np.mean(((validation_df - rec_NMF)**2).to_numpy()) 
    
   validation_tensor = torch.tensor(validation_df.values,
                                    dtype = torch.float32)
   rec_data = model_AE(validation_tensor)
   out_error_AE = loss_AE(validation_tensor, rec_data).detach().item()

   # cosine distance between NMF and AE sigs
   cos = cosine_HA(signatures_NMF, signatures_AE.T)[0]
   cos_mean = np.mean(cos.diagonal())

   #scale exposures so signatures sum to one
   diagonals_NMF = signatures_NMF.sum(axis = 1)
   scale_NMF = np.diag(diagonals_NMF)

   diagonals_AE = signatures_AE.sum(axis = 0)
   scale_AE = np.diag(diagonals_AE)


   exposures_NMF = exposures_NMF@scale_NMF
   
   exposures_AE = exposures_AE.T@scale_AE

   # exposure difference
   exp_diff = np.mean(((exposures_NMF - exposures_AE)**2).to_numpy())

   #out_error_difference
   our_err_diff = out_error_NMF - out_error_AE
   print(out_error_NMF)
   return cos_mean, exp_diff, our_err_diff

def performance_analysis(m, nsigs, epochs, i):
   mut_matrix = m.T.sample(200)
   train_data, test_data = split_data(mut_matrix, 0.8)
  
   train_data = train_data/train_data.max().max()
   test_data = test_data/train_data.max().max()


   res = compare_NMF_AE(train_data, test_data, nsigs, epochs)
   print(i)
   return(res)


n_sims = 50
n_sigs = 7
epochs = 1000
loss_name = "MSE"
optimizer_name = "Adam"


cwd = os.getcwd()
print(cwd)
res = Parallel(n_jobs = 2)(delayed(performance_analysis)(PCAWG_prost, n_sigs, epochs, i) for i in range(n_sims))
print("analysis_done")
result = pd.DataFrame(res)
result.columns = ["cosine", "exp", "error"]
print(result)
name = "compare"+ loss_name + "_ADAM_nsim:" + str(n_sims) + "Prost" + "_nsigs:" + str(n_sigs) + "_epochs:" + str(epochs)
#result.to_csv(name + '.csv')

#matplotlib.use('Agg')
fig=plt.figure()
# set height of each subplot as 8
fig.set_figheight(15)
 
# set width of each subplot as 8
fig.set_figwidth(15)

f, ax = plt.subplots()
points = ax.scatter(y = result['cosine'], x = result['error'], c = result['exp'] ,cmap="plasma")
#f.ylabel('mean diagonal cosine')
f.colorbar(points)
plt.show()

'''
plt.scatter(y = result['cosine'], x = result['error'], c = result['exp'] ,cmap="plasma")
plt.colorbar(points)
plt.ylabel('mean diagonal cosine')
plt.legend()
plt.show()
'''
#plt.savefig(name + "scatter.png", transparent=True)