#from tabnanny import verbose
import numpy as np
import torch
import pandas as pd

from sklearn import model_selection
from NMFAE_init import NMFAE, train_NMFAE
from functools import partial 
from skopt import space
from skopt import gp_minimize
from functions import simulate_counts


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

        train_NMFAE(epochs = 200, 
               model = model, 
               x_train = x_train, 
               loss_function = loss_function, 
               optimizer = optimizer,
               batch_size = 16)

        cv_test_tensor = torch.tensor(x_test.values, 
                                        dtype = torch.float32)

        cv_fit = model(cv_test_tensor)
        out_err.append(float(loss_function(cv_fit,cv_test_tensor).detach().numpy()))
    return np.mean(out_err)

X,_,_ = simulate_counts(5, 5000)
X = X.transpose()

param_space = [
    space.Integer(2,15, name = "nsig"),
    space.Real(0.000001, 1, prior = "uniform", name = "lr")
]

param_names = [
    "nsig", 
    "lr"
]


optimization_func = partial(
    optimize,
    param_names = param_names,
    X = X
)

result = gp_minimize(
    optimization_func, 
    dimensions = param_space,
    n_calls = 10,
    n_random_starts = 10,
    verbose = 10)

print(result.x)