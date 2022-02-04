#from tabnanny import verbose
import numpy as np
import torch
import pandas as pd

from sklearn import model_selection
from NMFAE_init import NMFAE, train_NMFAE
from functools import partial 
from skopt import space
from skopt import gp_minimize
from functions import simulate_counts, cosine_perm
from sklearn.decomposition import NMF



def optimize(params, param_names, X):
    params = dict(zip(param_names, params))
    model = NMFAE(params['nsig'])
    kf = model_selection.KFold()

    out_err = []
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])

    for train, test in kf.split(X):
        x_train = pd.DataFrame(X).iloc[train,:]
        x_test = pd.DataFrame(X).iloc[test,:]

        train_NMFAE(
            epochs = 200, 
            model = model, 
            x_train = x_train, 
            loss_function = loss_function, 
            optimizer = optimizer,
            batch_size = int(params['batch_size'])
            )

        cv_test_tensor = torch.tensor(x_test.values, 
                                        dtype = torch.float32)

        cv_fit = model(cv_test_tensor)
        out_err.append(float(loss_function(cv_fit,cv_test_tensor).detach().numpy()))
    return np.mean(out_err)


def out_errorNMF(params, param_names, X):
  params = dict(zip(param_names, params))
  model = NMF(n_components = params['nsig'], init='random', max_iter = 200)
  kf = model_selection.KFold()

  out_err = []
  for train, test in kf.split(X):
      x_train = pd.DataFrame(X).iloc[train,:]
      x_test = pd.DataFrame(X).iloc[test,:]
      exposures = model.fit_transform(x_train)
      signatures = model.components_

      ref_exposures = model.transform(X = x_test)
      rec = np.dot(ref_exposures, signatures)

      MSE = np.mean(((x_test - rec)**2).to_numpy())
      out_err.append(MSE)

  return(np.mean(out_err))


#X = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\DLBCL_1001\DLBCL_mut_matrix.tsv', sep='\t', index_col=0)

X,_,_ = simulate_counts(5, 2000)
X = X.transpose()

param_space = [
    space.Integer(2,15, name = "nsig"),
    space.Real(0.000001, 1, prior = "uniform", name = "lr"),
    space.Categorical([8, 16, 32, 64], name = 'batch_size')
]

param_names = [
    "nsig", 
    "lr",
    "batch_size"
]

#Partial functions allow us to fix a certain number of arguments of a function and generate a new function.
optimization_NMF = partial(
    out_errorNMF,
    param_names = param_names,
    X = X
)

optimization_AE = partial(
    optimize,
    param_names = param_names,
    X = X
)

result = []
for _ in range(5):
    resNMF = gp_minimize(
        optimization_NMF, 
        dimensions = param_space,
        n_calls = 15,
        n_random_starts = 10,
        verbose = 1)
    resAE = gp_minimize(
        optimization_AE, 
        dimensions = param_space,
        n_calls = 15,
        n_random_starts = 10,
        verbose = 1)
    result.append([resNMF.x, resAE.x])

print(result)