#from tabnanny import verbose
from tabnanny import verbose
import numpy as np
import torch
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn import model_selection
from AAUtoSig_init import AAUtoSig, train_AAUtoSig

import torch.optim as optim


#takes a data matrix nX96 and returns a dictionary of optimal hyperparameters
def optuna_tune(X, nsig):

    def objective(trial):
        #nsig = trial.suggest_int('nsig', 2, 15)
        lr = trial.suggest_float('lr',1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        
        
        #model = NMFAE(nsig)
        model = AAUtoSig(dim1 = nsig)
        kf = model_selection.KFold()

        out_err = []
        loss_function = torch.nn.MSELoss(reduction='mean')
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        for train, test in kf.split(X):
            x_train = pd.DataFrame(X).iloc[train,:]
            x_test = pd.DataFrame(X).iloc[test,:]
            '''
            train_NMFAE(
                epochs = 200, 
                model = model, 
                x_train = x_train, 
                loss_function = loss_function, 
                optimizer = optimizer,
                batch_size = int(batch_size)
                )
            '''
            train_AAUtoSig(
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
            err = float(loss_function(cv_fit,cv_test_tensor).detach().numpy())
            out_err.append(err)
        

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return np.mean(out_err)


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, timeout=600)

    trial = study.best_trial

    return trial.params
