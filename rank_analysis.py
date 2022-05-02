import numpy as np
import torch
import pandas as pd
import optuna
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from datetime import date

from sklearn import model_selection
from NMFAE_init import NMFAE, train_NMFAE
from functools import partial 
from skopt import space
from skopt import gp_minimize
from functions import simulate_counts
from sklearn.decomposition import NMF
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)


def MSE_NMF(X, nsig):
    model = NMF(n_components = nsig, init='random', max_iter = 200)
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

def skopt_NMF(X):
    def objective(params, param_names, X):
        params = dict(zip(param_names, params))
        res = MSE_NMF(X, params['nsig'])
        return res


    param_space = [space.Integer(2,15, name = "nsig")]

    param_names = ["nsig"]

    #Partial functions allow us to fix a certain number of arguments of a function and generate a new function.
    optimization_NMF = partial(
        objective,
        param_names = param_names,
        X = X
    )

    resNMF = gp_minimize(
        optimization_NMF, 
        dimensions = param_space,
        n_calls = 100,
        n_random_starts = 10,
        verbose = 0,
        n_jobs=3)

    return resNMF.x[0]
def optuna_NMF(X):

    def objective(trial):
        nsig = trial.suggest_int('nsig', 2, 15)
        res = MSE_NMF(X, nsig)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return np.mean(res)


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600, n_jobs=3)

    trial = study.best_trial

    res = trial.params
    return res['nsig']


def MSE_AE(X, nsig, lr, optimizer_name, batch_size):
    model = NMFAE(nsig)
    kf = model_selection.KFold()

    out_err = []
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)


    for train, test in kf.split(X):
        x_train = pd.DataFrame(X).iloc[train,:]
        x_test = pd.DataFrame(X).iloc[test,:]

        train_NMFAE(
            epochs = 200, 
            model = model, 
            x_train = x_train, 
            loss_function = loss_function, 
            optimizer = optimizer,
            batch_size = int(batch_size)
            )

        cv_test_tensor = torch.tensor(x_test.values, 
                                        dtype = torch.float32)

        cv_fit = model(cv_test_tensor)
        out_err.append(float(loss_function(cv_fit,cv_test_tensor).detach().numpy()))
    return np.mean(out_err)

#skopt_AE er uvenner mec SGD af en eller anden grund.
def skopt_AE(X):
    def optimize(params, param_names, X):
        params = dict(zip(param_names, params))
        lr = params['lr'] 
        nsig = params['nsig']
        optimizer_name = params['optimizer_name']
        batch_size = params['batch_size']
        res = MSE_AE(X, nsig, lr, optimizer_name, batch_size)
        return res

    param_space = [
        space.Integer(2, 15, name = "nsig"),
        space.Real(0.000001, 1, prior = "uniform", name = "lr"),
        space.Categorical([8, 16, 32, 64], name = 'batch_size'),
        space.Categorical( ["Adam", "RMSprop"], name = "optimizer_name")
    ]

    param_names = [
        "nsig", 
        "lr",
        "batch_size",
        "optimizer_name"
    ]

    optimization_AE = partial(
        optimize,
        param_names = param_names,
        X = X
    )

    resAE = gp_minimize(
        optimization_AE, 
        dimensions = param_space,
        n_calls = 100,
        n_random_starts = 10,
        verbose = 0,
        n_jobs=3)
    
    return resAE.x[0]

def optuna_AE(X):
    def objective(trial):
        nsig = trial.suggest_int('nsig', 2, 15)
        lr = trial.suggest_float('lr',1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        
        res = MSE_AE(X, nsig, lr, optimizer_name, batch_size)
        

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return res

    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 100, timeout = 600, n_jobs=3)

    trial = study.best_trial

    res = trial.params['nsig']
    #does this do anything?
    if math.isnan(res):
        res = 0
    return res

def rank_analysis(npatients, nsigs):
  X, _,_ = simulate_counts(nsigs, npatients)
  X = X.transpose()
  return(skopt_NMF(X), optuna_NMF(X), skopt_AE(X), optuna_AE(X))
asd = np.array([rank_analysis(200, 5) for _ in range(2)])
result = pd.DataFrame(asd)
result.columns = ["skopt_NMF", "optuna_NMF", "skopt_AE", "optuna_AE"]

outfile = "Q:\AUH-HAEM-FORSK-MutSigDLBCL222\article_1\output\rank_analysis" + str(date.today()) + ".png"

plt.hist(result, label  = result.columns)
plt.title("Number of signatures found in each tuning")
plt.legend()
plt.savefig(outfile)