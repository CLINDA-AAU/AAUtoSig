import argparse
import numpy as np
import pandas as pd
import itertools
import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
import cvxpy as cp
from functions import simulate_counts

''' Faktoriser datamatrix ud i abundances A og endpoints M 
    Et forward pass returnerer alle iterationer af M og A afhængig af de foregående iterationer af  M og A.
    Alle itrationer af M og A som lister, og initial values gives som argumenter _M _A
    M her svarer til A i artiklen og A her svarer til S i artiklen'''


class UnmixingUtils:
    def __init__(self, A, S):
        self.A = A
        self.S = S
        pass

    def hyperSAD(self, A_est):
        L = np.size(A_est, 0)
        R = np.size(A_est, 1)
        meanDistance = np.inf
        p = np.array(list(itertools.permutations(np.arange(0, R), R)))
        num = np.size(p, 0)
        sadMat = np.zeros([R, R])
        for i in range(R):
            temp = A_est[:, i]
            temp = np.tile(temp, (R, 1)).T
            sadMat[i, :] = np.arccos(
                sum(temp * self.A, 0) / (np.sqrt(np.sum(temp * temp, 0)+1e-4) * np.sqrt(np.sum(self.A * self.A, 0))+1e-4))
        sadMat = sadMat.T
        temp = np.zeros([1, R])
        sor = p[0, :]
        Distance = meanDistance
        for i in range(num):
            for j in range(R):
                temp[0, j] = sadMat[j, p[i, j]]
            if np.mean(temp) < meanDistance:
                sor = p[i, :]
                meanDistance = np.mean(temp, 1)
                Distance = temp
        return Distance, meanDistance, sor

    def hyperRMSE(self, S_est, sor):
        N = np.size(self.S, 0)
        S_est = S_est[:, sor]
        rmse = self.S - S_est
        rmse = rmse * rmse
        rmse = np.mean(np.sqrt(np.sum(rmse, 0) / N))
        return rmse


class RandomDataset(Dataset):
    def __init__(self, data, label, length):
        self.data = data
        self.len = length
        self.label = label

    #load two datasets in dataloader
    def __getitem__(self, item):
        return torch.Tensor(self.data[:,item]).float(), torch.Tensor(self.label[:,item]).float()

    def __len__(self):
        return self.len

class L1NMF_Net(nn.Module):
    def __init__(self, layerNum, M, A):
        super(L1NMF_Net, self).__init__()
        self.A = A
        self.M = M
        #I think this part is just parameterinitialization
        R = np.size(M, 1)
        eig, _ = np.linalg.eig(M.T @ M)
        eig += 0.1
        L = 1 / np.max(eig)
        theta = np.ones((1, R)) * 0.01 * L
        # Endmember
        eig, _ = np.linalg.eig(A @ A.T)
        eig += 0.1
        L2 = np.max(eig)
        L2 = 1 / L2

        self.p = nn.ParameterList()
        self.L = nn.ParameterList() #t_2k
        self.theta = nn.ParameterList()
        self.L2 = nn.ParameterList() #t_1k
        self.W_a = nn.ParameterList() 
        self.layerNum = layerNum
        temp = self.calW(M)
        for k in range(self.layerNum):
            self.L.append(nn.Parameter(torch.FloatTensor([L])))
            self.L2.append(nn.Parameter(torch.FloatTensor([L2])))
            self.theta.append(nn.Parameter(torch.FloatTensor(theta)))
            self.p.append(nn.Parameter(torch.FloatTensor([0.5])))
            self.W_a.append(nn.Parameter(torch.FloatTensor(temp)))
        self.layerNum = layerNum
    def forward(self, X):
        _M = self.M
        _A = self.A
        self.W_m = torch.FloatTensor(_A.values)
        M = list()
        M.append(torch.FloatTensor(_M.values))
        A = list()
        A.append(torch.FloatTensor(_A.values.T))
        for k in range(self.layerNum):
            theta = self.theta[k].repeat(A[-1].size(1), 1).T
            T = (M[-1].mm(A[-1]) - X).float()
            #Input to GST
            _A = A[-1] - self.L[k]*self.W_a[k].T.mm(T)
            #equation 11
            _A = self.sum2one(F.relu(self.self_active(_A, self.p[k], theta)))
            A.append(_A)
            #equation 9
            T = (M[-1].mm(A[-1]) - X).float()
            _M = M[-1] - T.mm(self.L2[k] * self.W_m) #W_m is tensor of _A
            _M = F.relu(_M)
            M.append(_M)
        return M, A
    def self_active(self, x, p, lam):
        #equation 14
        tau=pow(2*(1-p)*lam,1/(2-p))+p*lam*pow(2*lam*(1-p), (p-1)/(2-p))
        v = x
        ind = (x-tau) > 0
        ind2=(x-tau)<=0
        #equation 13 and 15
        v[ind]=x[ind].sign() * (x[ind].abs() - p * lam[ind] * pow(x[ind].abs(), p - 1))
        v[ind2]=0
        v[v>1]=1
        return v
    def calW(self,D):
        (m,n)=D.shape
        W = cp.Variable(shape=(m, n))
        obj = cp.Minimize(cp.norm(W.T @ D, 'fro'))
        # Create two constraints.
        constraint = [cp.diag(W.T @ D) == 1]
        prob = cp.Problem(obj, constraint)
        result = prob.solve(solver=cp.SCS, max_iters=1000)
        print('residual norm {}'.format(prob.value))
        # print(W.value)
        return W.value
    def sum2one(self, Z):
        temp = Z.sum(0)
        temp = temp.repeat(Z.size(0), 1) + 0.0001
        return Z / temp



#I think this model depends on the number of observations being divisible by batch size

data, sigs, exp = simulate_counts(10, 600)

#net = L1NMF_Net(2, M = exp.transpose(), A = sigs)
#print(net)

data = data.transpose().values
'''
a,b = net(data)
print(a[-1].size())
print(b[-1].size())
'''

#using the net on data returns an exposure and signature matrix.

def train(lrD,layerNum, lr, train_data, nrtrain, A0, S0, X, A, s):
    batch_size = nrtrain
    #args = set_param(layerNum, lr, lrD,batch_size=batch_size)
    model = L1NMF_Net(layerNum, A0, S0)
    criterion = nn.MSELoss(reduction='sum')
    trainloader = DataLoader(dataset=RandomDataset(train_data, S0.to_numpy().T, nrtrain), batch_size = batch_size,
                             num_workers=0,
                             shuffle=False)
    learning_rate = lr
    learning_rate_decoder= lrD
    opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                       model.theta],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))

    running_loss = 0.0
    last_loss=1
    for epoch_i in range(100):

        for data_batch in trainloader:
            #batch_label er abundences
            batch_x, batch_label = data_batch
            output_end, output_abun= model(batch_x.T)#,A0,batch_label)
            #output_end og output_abun er lister. Denne laver loss over alle iterationer af listen
            loss=sum([criterion(output_end[i+1] @ output_abun[i+1], batch_x.T) for i in range(layerNum)])/layerNum
            opt.zero_grad()
            loss.backward()
            opt.step()
        for i in range(layerNum):
            t1 = model.p[i].data
            t1[t1 < 0] = 1e-4
            t1[t1 > 1] = 1
            model.p[i].data.copy_(t1)
            running_loss += loss.item()
        temp = abs(running_loss - last_loss) / last_loss
        output_data = 'train===epoch: %d, loss:  %.5f, tol: %.6f\n' % (epoch_i, running_loss, temp)
        print(output_data)
        last_loss=running_loss
        running_loss = 0.0

    util = UnmixingUtils(A, s.T)
    out1, out2 = model(torch.FloatTensor(X))#, A0, S0.T)
    Distance, meanDistance, sor = util.hyperSAD(out1[-1].detach().numpy())
    rmse = util.hyperRMSE(out2[-1].T.detach().numpy(), sor)
    output_data = 'Res: SAD: %.5f RMSE:  %.5f' % (meanDistance, rmse)
    print(output_data)
    return meanDistance,rmse

layerNum =9
lr = 2
lrD = 1e-6
# For SNR=15
# lr = 0.3
# lrD = 1e-8
train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=data.transpose(),  nrtrain=600, A0=sigs, S0=exp.T,
        X=data, A=sigs, s=exp.T)
