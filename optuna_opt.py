#from tabnanny import verbose
from tabnanny import verbose
import numpy as np
import torch
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn import model_selection
from AAUtoSig_init import AAUtoSig, train_AAUtoSig
from NMFAE_init import NMFAE, train_NMFAE
from functions import simulate_counts



#takes a data matrix nX96 and returns a dictionary of optimal hyperparameters
def optuna_tune(X, nsig, criterion, optimizer_alg):

    def objective(trial):
        #nsig = trial.suggest_int('nsig', 2, 15)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        lr = trial.suggest_float('lr',1e-8, 1e-1, log=True)
        
        model = AAUtoSig(dim1 = nsig)
        if optimizer_alg == "Adam":
          optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        if optimizer_alg == "Tuned":
          optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
          optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
        kf = model_selection.KFold()

        out_err = []
        loss_function = criterion

        
        for train, test in kf.split(X):
            x_train = pd.DataFrame(X).iloc[train,:]
            x_test = pd.DataFrame(X).iloc[test,:] 
            train_AAUtoSig(
                epochs = 1000, 
                model = model, 
                x_train = x_train, 
                loss_function = loss_function, 
                optimizer = optimizer,
                batch_size = int(batch_size)
                )
            
            cv_test_tensor = torch.tensor(x_test.values, 
                                            dtype = torch.float32)

            cv_fit = model(cv_test_tensor)
            
            err = float(loss_function(cv_fit,cv_test_tensor).detach().numpy())
            out_err.append(err)
        

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return np.mean(out_err)

    
    study = optuna.create_study(direction="minimize")
    
    study.optimize(objective, n_trials=20, timeout=600) 
    trial = study.best_trial

    return trial.params
'''
nsigs = 5

mf_df, true_sigs,_ = simulate_counts(nsigs, 100, pentanucelotide = False)
trinucleotide = mf_df.index
mutation = [t[2:5] for t in trinucleotide]

X = mf_df.T

model = NMFAE(dim1 = nsigs)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss(reduction='mean')

asd = optuna_tune(X, 5, criterion = torch.nn.MSELoss(reduction='mean'), optimizer_alg = "Tuned")
print(asd)
'''
