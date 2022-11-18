
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

from functions import cosine_HA, split_data
from AAUtoSig_init import AAUtoSig, train_AAUtoSig
import optuna
from optuna_opt import optuna_tune

from joblib import Parallel, delayed
import multiprocessing

#os.chdir("../..")
os.chdir(r"dfs_forskning/AUH-HAEM-FORSK-MutSigDLBCL222/article_1")

#m = pd.read_csv(r"generated_data/SomaticSiMu/Lung-SCC_SBS96.csv")#.drop(['Unnamed: 0'], axis=1)
m = pd.read_csv(r"generated_data/SomaticSiMu/Ovary_SBS96.csv")


#groundtruth = [line.strip("\n") for line in open(r"generated_data/SomaticSiMu/Lung-SCC_sbs_sigs.txt")]
groundtruth = [line.strip("\n") for line in open(r"generated_data/SomaticSiMu/Ovary-AdenoCA_sbs_sigs.txt")]

groundtruth = pd.DataFrame([ast.literal_eval(x) for x in groundtruth])
groundtruth[0] = groundtruth[0].astype(int)

os.chdir(r"scripts/AAUtoSig")


sig_names = np.unique(groundtruth.drop([0], axis = 1).fillna(":("))[1::]

COSMIC = pd.read_csv("COSMIC/COSMIC_v3.2_SBS_GRCh37.txt", sep = '\t', index_col=0)
sigs = COSMIC[sig_names]
  

def out_errorNMF(train_df, validation_df, nsigs, true_sigs, criterion):
   if criterion == "KL" or criterion == "PoNNL":
      beta = 1
   if criterion == "MSE":
      beta = 2
   
   model = NMF(n_components=nsigs, init='random', max_iter = 5000, solver='mu', beta_loss = beta)
   exposures = model.fit_transform(train_df)
   signatures = model.components_
   cos_NMF = cosine_HA(signatures, true_sigs.to_numpy().T)[0]
   cos_mean = np.mean(cos_NMF.diagonal())

   ref_exposures = model.transform(X = validation_df)
   rec = np.dot(ref_exposures, signatures)
   if criterion == "KL" or criterion == "PoNNL":
      res = scipy.special.kl_div(validation_df, rec)
      out_error = res.mean().mean()
   if criterion == "MSE":
      out_error = np.mean(((validation_df - rec)**2).to_numpy())
  
   print(out_error)
   return cos_mean, out_error
  
def out_error_AAUtoSig(train_df, validation_df, nsigs, true_sigs, loss_name, optimizer_alg, i):
   model = AAUtoSig(nsigs)


   params = optuna_tune(train_df, nsigs, loss_name, optimizer_alg)
   if optimizer_alg == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])
   if optimizer_alg == "Tuned":
      optimizer =  getattr(torch.optim, params['optimizer'])(model.parameters(), lr = params['lr'])
   _, out_error, _ = train_AAUtoSig(5000, model, train_df, validation_df, loss_name= loss_name, optimizer=optimizer, batch_size = params['batch_size'], do_plot = True, ES = False, i = i)

   signatures = model.dec1.weight.data    
   signatures = pd.DataFrame(signatures.numpy())

   if any(np.sum(signatures, axis = 0) == 0):
      #print(np.sum(signatures, axis = 0) )
      #print(signatures.loc[:, np.sum(signatures, axis = 0) == 0] )
      signatures.loc[:, np.sum(signatures, axis = 0) == 0] += 1e-5
      #print(np.sum(signatures, axis = 0) )
   Y = cosine_HA(signatures.T, true_sigs.T)
   
   cos_mean = np.mean(Y[0].diagonal())
   return cos_mean, out_error, i
    
  
def performance_analysis(m, m_sigs, COSMIC, criterion, optimizer_alg, i):
   m = m.sort_values('0') #hvorfor gør jeg nu det her - fordi det også er sådan signaturerne er organiseret
   mut_matrix = (m.drop(['Unnamed: 0', '0'], axis = 1)).T.sample(1500)
   train_data, test_data = split_data(mut_matrix, 0.8)
  
   idx_train = train_data.index
   idx_test = test_data.index
   y = [str(x) for x in m_sigs[0]]

   train_idx = list (set(idx_train) & set(y))
   test_idx = list(set(idx_test) & set(y))
   train_data = train_data[train_data.index.isin(train_idx)]/train_data.max().max()
   test_data = test_data[test_data.index.isin(test_idx)]/train_data.max().max()

   m_sigs_train = m_sigs[m_sigs[0].isin([int(x) for x in train_idx])]
   m_sigs_test = m_sigs[m_sigs[0].isin([int(x) for x in test_idx])]
   sig_names_train = np.unique(m_sigs_train.drop([0], axis = 1).fillna(":("))[1::]
   sig_names_test = np.unique(m_sigs_test.drop([0], axis = 1).fillna(":("))[1::]
   
   nsigs = len(sig_names_train)
   #COSMIC = pd.read_csv(r'COSMIC\COSMIC_v3.1_SBS_GRCh38.txt', sep = '\t', index_col=0)
   #old = COSMIC['SBS1']
   #Arrange COSMIC to be the same ordering as count data
   COSMIC = COSMIC.sort_index()

   sigs_train = COSMIC[sig_names_train]

   res_NMF = out_errorNMF(train_data, test_data, nsigs, sigs_train, criterion)
   
   res_AE = out_error_AAUtoSig(train_data, test_data, nsigs, sigs_train, criterion, optimizer_alg, i)
   return(res_NMF + res_AE)

n_sims = 50
optimizer_alg = "MSE"
#data = (m.drop(['Unnamed: 0', '0'], axis = 1)).T



res = Parallel(n_jobs = 10)(delayed(performance_analysis)(m, groundtruth, COSMIC, optimizer_alg, "Adam", i) for i in range(n_sims))
result = pd.DataFrame(res)
result.columns = ["NMF_perm", "outNMF", "AE_perm", "outAE" , "idx"]
result.to_csv('result.csv')
print(result)
name = "Lung_SCC_"+ optimizer_alg + "_ADAM_nsim:" + str(n_sims)


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


#params = [("MSE", "Adam"), ("KL", "Adam"), ("MSE", "Tuned")]

#Parallel(n_jobs = 3)(delayed(analysis_to_plot)(m, groundtruth, COSMIC, p[0], p[1], 3) for p in params)

#analysis_to_plot(m, groundtruth, COSMIC, "MSE", "Adam", 1)
