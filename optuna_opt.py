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
from EGD import EGD_init, train_EGD
#from egpm import EGPM
from EGD_optimizer import EGD_optim

import torch.optim as optim


#takes a data matrix nX96 and returns a dictionary of optimal hyperparameters
def optuna_tune(X, nsig, model_name = "NMFAE"):

    def objective(trial):
        #nsig = trial.suggest_int('nsig', 2, 15)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        #lr1 = trial.suggest_float('lr1',1e-8, 1e-1, log=True)

        if model_name == "NMFAE":
            model = NMFAE(nsig)
        if model_name == "AAUtoSig":
            model = AAUtoSig(dim1 = nsig)
        if model_name == "EGD":
            lr2 = trial.suggest_float('lr2',1e-2, 1000, log=True)
            model = EGD_init(hidden_dim = nsig, input_dim = 1024)
            #optimizer_enc = torch.optim.Adam(model.enc1.parameters(), lr = lr1)
            optimizer_dec = EGD_optim(model.parameters(), lr = lr2)
            #optimizer_dec = EGPM(model.dec1.parameters(), lr = lr2, u_scaling=1,
                             #norm_per=None, gradient_clipping=True, 
                             #weight_regularization=None, plus_minus=False,
                             #init='bootstrap')
        
        if model_name == "NMFAE" or model_name == "AAUtoSig":
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)#lr1)
        kf = model_selection.KFold()

        out_err = []
        loss_function = torch.nn.MSELoss(reduction='mean')


        for train, test in kf.split(X):
            x_train = pd.DataFrame(X).iloc[train,:]
            x_test = pd.DataFrame(X).iloc[test,:]
            if model_name == "NMFAE":
                train_NMFAE(
                    epochs = 2000, 
                    model = model, 
                    x_train = x_train, 
                    loss_function = loss_function, 
                    optimizer = optimizer,
                    batch_size = int(batch_size)
                    )
            if model_name == "AAUtoSig":    
                train_AAUtoSig(
                    epochs = 2000, 
                    model = model, 
                    x_train = x_train, 
                    loss_function = loss_function, 
                    optimizer = optimizer,
                    batch_size = int(batch_size)
                    )
            if model_name == "EGD":
                train_EGD(epochs = 2000, 
                    model = model, 
                    x_train = x_train, 
                    loss_function = loss_function, 
                    #optimizer_enc = optimizer_enc,
                    optimizer = optimizer_dec,
                    batch_size = int(batch_size))

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
