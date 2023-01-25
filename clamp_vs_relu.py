import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from matplotlib import gridspec
from joblib import Parallel, delayed


from AAUtoSig_init import AAUtoSig, train_AAUtoSig
from functions import simulate_counts, plotsigs, cosine_HA, split_data
from optuna_opt import optuna_tune

#ReLU non-negativity is 'enforced' in the model definition - PG non negativity is enforced in training



# implement multiplicative updates

#out_error performance
def out_error_AAUtoSig(train_df, validation_df, model, true_sigs, loss, epochs, non_negativity):
    params = optuna_tune(train_df, model, loss, non_negativity = non_negativity)   
    optimizer = torch.optim.Adam(model.parameters(),
                            lr = params['lr'])

    _, out_error, _, signatures, _, _ = train_AAUtoSig(epochs, 
                                          model, 
                                          train_df, 
                                          validation_df, 
                                          criterion = loss, 
                                          optimizer = optimizer, 
                                          batch_size = params['batch_size'], 
                                          do_plot = False, 
                                          ES = False, 
                                          non_negative = non_negativity)
                                        
    #asd1 = model.dec1.weight.data
    #asd = list(model.dec1.parameters())[0]
    
    #print(sum(asd != asd))


    if  non_negativity == "None":
      signatures = np.clip(model.dec1.weight.data, a_min=0, a_max = None)
    signatures = pd.DataFrame(signatures.numpy())

    if np.any(np.isinf(signatures) | np.isnan(signatures)): 
      print("hopla - hvad sker der lige her?")
      print(signatures)
    if np.any((signatures.T==0).all(axis=1)):
      print(signatures.iloc[(signatures==0).all(axis=0),:])
      signatures.iloc[(signatures==0).all(axis=0),:] += 1e-3
    cos_AE = cosine_HA(signatures, true_sigs.T)[0]
    cos_mean = np.mean(cos_AE.diagonal())
      
    return cos_mean, out_error


def performance_analysis(npatients, nsigs, loss, epochs):
  mut_matrix, signatures, _ = simulate_counts(nsigs, npatients)
  train_data, validation_data = split_data(mut_matrix.T, 0.8)
  train_data = train_data/train_data.max().max()
  validation_data = validation_data/train_data.max().max()
  
  #projected gradient model
  m_pg = AAUtoSig(96, nsigs, relu_activation = [False, False])

  #relu model
  m_relu = AAUtoSig(96, nsigs, relu_activation =[True, True])

  #relu encoder PG decoder (as NSAE)
  m_comb = AAUtoSig(96, nsigs, relu_activation = [True, False])
  
  cosineAE_ReLU, out_ReLU = out_error_AAUtoSig(train_data, validation_data, m_relu, signatures, loss = loss, epochs = epochs, non_negativity = "None")
  cosineAE_PG, out_PG = out_error_AAUtoSig(train_data, validation_data, m_pg, signatures, loss = loss, epochs = epochs, non_negativity = "all")
  cosineAE_comb, out_comb = out_error_AAUtoSig(train_data, validation_data, m_comb, signatures, loss = loss, epochs = epochs, non_negativity = "bases")

  return [cosineAE_ReLU, out_ReLU, cosineAE_PG, out_PG, cosineAE_comb, out_comb]  

#use and simulated data and save plots
n_sims = 50
npatients = 3000
nsigs = 7
epochs = 5000
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss(reduction='mean')


# Using an Adam Optimizer with lr = 1e-3
#optimizer_pg = torch.optim.Adam(m_pg.parameters(),
#                            lr = 1e-3)
#optimizer_relu = torch.optim.Adam(m_relu.parameters(),
#                            lr = 1e-3)
#optimizer_comb = torch.optim.Adam(m_comb.parameters(),
#                            lr = 1e-3)

res = Parallel(5)(delayed(performance_analysis)(npatients, nsigs, loss_function, epochs) for i in range(n_sims))
result = pd.DataFrame(res)
result.columns = ["ReLU_perm", "outReLU", "PG_perm", "outPG", "comb_perm", "outcomb" ]
print(result)

matplotlib.use('Agg')
fig=plt.figure()
# set height of each subplot as 8
fig.set_figheight(15)
 
# set width of each subplot as 8
fig.set_figwidth(15)

spec = gridspec.GridSpec(ncols=2, nrows=2,
                         width_ratios=[2.3, 1], wspace=0.2,
                         hspace=0.2, height_ratios=[2.3, 1])


ax1 = fig.add_subplot(spec[0])
plt.scatter(y = result['ReLU_perm'], x = result['outReLU'], c = 'blue', label = 'ReLU')
plt.scatter(y = result['PG_perm'], x = result['outPG'], c = 'red', label = 'PG')
plt.scatter(y = result['comb_perm'], x = result['outcomb'], c = 'green', label = 'ReLU + PG')
#plt.xlabel('Out of sample error')
plt.ylabel('mean diagonal cosine')
plt.legend()
ax2 = fig.add_subplot(spec[1])
plt.boxplot(result[['ReLU_perm', 'PG_perm', 'comb_perm']] ,labels = ["ReLU", "PG", "ReLU + PG"])
#plt.ylabel('mean diagonal cosine')
ax3 = fig.add_subplot(spec[2])
plt.boxplot(result[['outReLU', 'outPG', 'outcomb']], labels = ["ReLU", "PG", "ReLU + PG"], vert=False)
plt.xlabel('Out of sample error')

ax1.get_shared_x_axes().join(ax1, ax3)
ax1.get_shared_y_axes().join(ax1, ax2)

plt.savefig("non_negativities.png", transparent=True)
