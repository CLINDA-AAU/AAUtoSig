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

#projected gradient model
m_pg = AAUtoSig(96, 5, non_negativity = None)

#relu model
m_relu = AAUtoSig(96, 5, non_negativity = "ReLU")

#relu encoder PG decoder (as NSAE)
class AAUtoSig_comb(torch.nn.Module):
    def __init__(self, feature_dim, latent_dim):
    
        super().__init__()
        

        # Building an linear encoder
        # 96 => dim
        self.enc1 = torch.nn.Linear(feature_dim, latent_dim, bias = False)
          
        # Building an linear decoder 
        # dim ==> 96
        self.dec1 = torch.nn.Linear(latent_dim, feature_dim, bias = False)
            

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.dec1(x)
        return x

m_comb = AAUtoSig_comb(96, 5)


# implement multiplicative updates

#out_error performance
def out_error_AAUtoSig(train_df, validation_df, model, true_sigs, loss, optimizer, epochs, non_negativity):
    train = train_df.columns
    sigs = true_sigs.index 
    #params = optuna_tune(train_df, nsigs, loss, optimizer)   

    _, out_error, _ = train_AAUtoSig(epochs, 
                                      model, 
                                      train_df, 
                                      validation_df, 
                                      loss_name= loss, 
                                      optimizer=optimizer, 
                                      batch_size = 32, 
                                      do_plot = False, 
                                      ES = False, 
                                      non_negative = non_negativity)

    signatures = model.dec1.weight.data    
    signatures = pd.DataFrame(signatures.numpy())

    cos_AE = cosine_HA(signatures.T, true_sigs.T)[0]
    cos_mean = np.mean(cos_AE.diagonal())
      
    return cos_mean, out_error


def performance_analysis(npatients, nsigs, loss, epochs):
  mut_matrix, signatures, _ = simulate_counts(nsigs, npatients)
  train_data, validation_data = split_data(mut_matrix.T, 0.8)
  train_data = train_data/train_data.max().max()
  validation_data = validation_data/train_data.max().max()
  cosineAE_ReLU, out_ReLU = out_error_AAUtoSig(train_data, validation_data, m_relu, signatures, loss = loss, optimizer = optimizer_relu, epochs = epochs)
  cosineAE_PG, out_PG = out_error_AAUtoSig(train_data, validation_data, m_pg, signatures, loss = loss, optimizer_name = optimizer_pg, epochs = epochs)
  cosineAE_comb, out_comb = out_error_AAUtoSig(train_data, validation_data, m_comb, signatures, loss = loss, optimizer_name = optimizer_comb, epochs = epochs)

  return [cosineAE_ReLU, out_ReLU, cosineAE_PG, out_PG, cosineAE_comb, out_comb]  

#use and simulated data and save plots
n_sims = 50
n_patients = 6000
n_sigs = 7
epochs = 5000
loss_function = torch.nn.MSELoss(reduction='mean')

res = Parallel(n_jobs = 3)(delayed(performance_analysis)(n_patients, n_sigs, loss_function, epochs) for i in range(n_sims))
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
plt.boxplot(result[['NMF_perm', 'AE_perm']] ,labels = ["NMF", "AE"])
#plt.ylabel('mean diagonal cosine')
ax3 = fig.add_subplot(spec[2])
plt.boxplot(result[['outAE', 'outNMF']], labels = ["AE", "NMF"], vert=False)
plt.xlabel('Out of sample error')

ax1.get_shared_x_axes().join(ax1, ax3)
ax1.get_shared_y_axes().join(ax1, ax2)

plt.savefig("non_negativities.png", transparent=True)
