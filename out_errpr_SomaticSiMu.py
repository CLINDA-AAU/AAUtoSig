
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy 

import numpy as np
import ast
import torch
from sklearn.decomposition import NMF

from random import sample

from functions import cosine_HA, simulate_counts
from AAUtoSig_init import AAUtoSig, train_AAUtoSig
import optuna
from optuna_opt import optuna_tune

from joblib import Parallel, delayed
import multiprocessing

os.chdir("../..")
os.chdir(r"dfs_forskning/AUH-HAEM-FORSK-MutSigDLBCL222/article_1")
'''
m = pd.read_csv(r"generated_data/SomaticSiMu/Lung-SCC_SBS96.csv")#.drop(['Unnamed: 0'], axis=1)

groundtruth = [line.strip("\n") for line in open(r"generated_data/SomaticSiMu/Lung-SCC_sbs_sigs.txt")]
groundtruth = pd.DataFrame([ast.literal_eval(x) for x in groundtruth])
groundtruth[0] = groundtruth[0].astype(int)

os.chdir(r"scripts/AAUtoSig")


sig_names = np.unique(groundtruth.drop([0], axis = 1).fillna(":("))[1::]
'''
COSMIC = pd.read_csv("COSMIC/COSMIC_v3.2_SBS_GRCh37.txt", sep = '\t', index_col=0)
#sigs = COSMIC[sig_names]

def split_data(data, frac_1):
  x_train = data.sample(frac = frac_1)
  x_validation = data.drop(x_train.index)
  return x_train, x_validation
  

def out_errorNMF(train_df, validation_df, nsigs, true_sigs, criterion):
   if criterion == "KL":
      beta = 1
   if criterion == "MSE":
      beta = 2
   
   model = NMF(n_components=nsigs, init='random', max_iter = 5000, solver='mu', beta_loss = beta)
   exposures = model.fit_transform(train_df)
   signatures = model.components_
   cos_NMF = cosine_HA(signatures, true_sigs.to_numpy().T)[0]
   cos_mean = np.mean(cos_NMF.diagonal())
   #cos_max = cosine_max(signatures, true_sigs.to_numpy().T)

   ref_exposures = model.transform(X = validation_df)
   rec = np.dot(ref_exposures, signatures)
   if criterion == "KL":
      res = scipy.special.kl_div(validation_df, rec)
      out_error = res.mean().mean()
   if criterion == "MSE":
      out_error = np.mean(((validation_df - rec)**2).to_numpy())
  
   print(out_error)
   return cos_mean, out_error
  
def out_error_AAUtoSig(train_df, validation_df, nsigs, true_sigs, criterion, optimizer_alg):
   model = AAUtoSig(nsigs)

   # Validation using MSE Loss function
   loss = criterion

   params = optuna_tune(train_df, nsigs, criterion, optimizer_alg)
   # Using an Adam Optimizer with lr = 1e-3
   if optimizer_alg == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])
   if optimizer_alg == "Tuned":
      optimizer =  getattr(torch.optim, params['optimizer'])(model.parameters(), lr = params['lr'])
   train_AAUtoSig(5000, model, train_df, loss_function= loss, optimizer=optimizer, batch_size = params['batch_size'])

   signatures = model.dec1.weight.data    
   signatures = pd.DataFrame(signatures.numpy())
   validation_tensor = torch.tensor(validation_df.values,
                                    dtype = torch.float32)
   rec_data = model(validation_tensor)
   out_error = criterion(validation_tensor, rec_data).detach().item()
   #MSE = np.mean(((validation_df - rec_data.detach().numpy())**2).to_numpy())
   #return(MSE)
   
   if any(np.sum(signatures, axis = 0) == 0):
      #print(np.sum(signatures, axis = 0) )
      #print(signatures.loc[:, np.sum(signatures, axis = 0) == 0] )
      signatures.loc[:, np.sum(signatures, axis = 0) == 0] += 1e-5
      #print(np.sum(signatures, axis = 0) )
   Y = cosine_HA(signatures.T, true_sigs.T)
   
   cos_mean = np.mean(Y[0].diagonal())
   return cos_mean, out_error
    
  
def performance_analysis(m, signatures, COSMIC, criterion, optimizer_alg ):#m_sigs, COSMIC, criterion, optimizer_alg):
   mut_matrix = m.sample(600)
   #m = m.sort_values('0')
   #mut_matrix = (m.drop(['Unnamed: 0', '0'], axis = 1)).T.sample(400)
   train_data, test_data = split_data(mut_matrix, 0.8)
  
   idx_train = train_data.index
   idx_test = test_data.index
   #y = [str(x) for x in m_sigs[0]]

   #train_idx = list (set(idx_train) & set(y))
   #test_idx = list(set(idx_test) & set(y))
   #train_data = train_data[train_data.index.isin(train_idx)]/train_data.max().max()
   #test_data = test_data[test_data.index.isin(test_idx)]/train_data.max().max()

   #m_sigs_train = m_sigs[m_sigs[0].isin([int(x) for x in train_idx])]
   #m_sigs_test = m_sigs[m_sigs[0].isin([int(x) for x in test_idx])]
   #sig_names_train = np.unique(m_sigs_train.drop([0], axis = 1).fillna(":("))[1::]
   #sig_names_test = np.unique(m_sigs_test.drop([0], axis = 1).fillna(":("))[1::]
   
   nsigs = 6 #len(sig_names_train)
   #COSMIC = pd.read_csv(r'COSMIC\COSMIC_v3.1_SBS_GRCh38.txt', sep = '\t', index_col=0)
   #old = COSMIC['SBS1']
   #Arrange COSMIC to be the same ordering as count data
   COSMIC = COSMIC.sort_index()

   #sigs_train = COSMIC[sig_names_train]

   res_NMF = out_errorNMF(train_data, test_data, nsigs,  signatures, criterion)#sigs_train, criterion)

   if criterion == "KL":
      criterion_AE = torch.nn.KLDivLoss(reduction = 'batchmean')
   if criterion == "MSE":
      criterion_AE = torch.nn.MSELoss()
   
   res_AE = out_error_AAUtoSig(train_data, test_data, nsigs, signatures, criterion_AE, optimizer_alg)# sigs_train, criterion_AE, optimizer_alg)
   return(res_NMF + res_AE)



#data = (m.drop(['Unnamed: 0', '0'], axis = 1)).T
def analysis_to_plot(data, true_sigs, library, criterion, optimizer_alg, n_sims):
   res = [performance_analysis(data, true_sigs, library, criterion, optimizer_alg) for _ in range(n_sims)]   
   result = pd.DataFrame(res)
   #print(res)
   result.columns = ["NMF_perm", "outNMF", "AE_perm", "outAE" ]
   
   
   #os.chdir("..")
   
   #name = "Lung_SCC_" + str(criterion) + "_" + str(optimizer_alg) + "_nsim:" + str(n_sims)
   name = "linear" + str(criterion) + "_" + str(optimizer_alg) + "_nsim:" + str(n_sims)
   matplotlib.use('Agg')
   plt.boxplot(result[['outNMF', 'outAE']], labels = ["NMF", "AE"])
   plt.title('NMF vs AE out of sample error on linearly simulated data')
   plt.savefig(name + "_error.png", transparent=True)
   plt.clf()
   
   
   plt.boxplot(result[['NMF_perm', 'AE_perm']] ,labels = ["NMF", "AE"])
   plt.title('NMF vs AE cosine values on SomaticSiMu simulated data')
   plt.savefig(name + "_cosine.png", transparent=True)
   plt.clf()
   
   plt.scatter(y = result['NMF_perm'], x = result['outNMF'], c = 'blue', label = 'NMF')
   plt.scatter(y = result['AE_perm'], x = result['outAE'], c = 'red', label = 'AE')
   plt.xlabel('Out of sample error')
   plt.ylabel('mean diagonal cosine')
   plt.title('NMF vs AE performance on SomaticSiMu simulated data')
   plt.legend()
   plt.savefig(name + "scatter.png", transparent=True)
   os.chdir("../../../../..")


nsigs = 6

mf_df, true_sigs,_ = simulate_counts(nsigs, 1000, pentanucelotide = False)
trinucleotide = mf_df.index
mutation = [t[2:5] for t in trinucleotide]

X = mf_df.T
print(true_sigs.shape)

params = [("MSE", "Adam"), ("KL", "Adam"), ("MSE", "Tuned")]

Parallel(n_jobs = 10)(delayed(analysis_to_plot)(X, true_sigs, COSMIC, p[0], p[1], 50) for p in params)

#Parallel(n_jobs = 10)(delayed(analysis_to_plot)(m, groundtruth, COSMIC, p[0], p[1], 50) for p in params)

#analysis_to_plot(m, groundtruth, COSMIC, "MSE", "Adam", 1)
