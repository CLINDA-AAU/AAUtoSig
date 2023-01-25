import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from AAUtoSig_init import AAUtoSig, train_AAUtoSig
import torch
from sklearn.decomposition import NMF
# ---------------------- CONVEX NMF algorithm -------------------------
def convex_nmf(X, rank, iter):
    n = X.shape[1]
    kmeans = KMeans(n_clusters=rank).fit(X.T)

    #The Kmeans labeling does not make sense when X is used the observations as rows
    print(kmeans.labels_)
    H = pd.get_dummies(kmeans.labels_) #n*k
    n_vec = H.sum(axis = 0)
    G = H + 0.2*np.ones((n, rank)) #n*k
    W = G@np.diag(1/np.array(n_vec)) #n*k  
    W = W.to_numpy()
    G = G.to_numpy()
    XtX = X.T@X

    for _ in range(iter):
        XtXW = XtX@W
        XtXG = XtX@G
        GWtXtXW = G@W.T@XtXW
        XtXWGtG = XtXW@G.T@G
        G = G * np.sqrt(XtXW/GWtXtXW)
        W = W * np.sqrt(XtXG/XtXWGtG)
    return(G, W)
'''
fig1, axs1 = plt.subplots(3,2, width_ratios = [4,1])
for i,n in enumerate([10, 50, 100]):

# ---------------- Data Initialization ------------------------
    signature1 = np.ones(6)/6
    signature2 = np.array([0]*3 + [2]*3)/6
    total_n = 3*n
    exposures = np.array([18,2]*n + [10,10]*n + [2,18]*n).reshape(total_n,2)
    XMat1 = np.random.poisson(18*signature1 + 2*signature2, size = (n,6))
    XMat2 = np.random.poisson(10*signature1 + 10*signature2, size = (n,6))
    XMat3 = np.random.poisson(2*signature1 + 18*signature2, size = (n,6))
    XMat = np.concatenate((XMat1, XMat2, XMat3))
    n_iter = 500

    # -------------------------- CVX NMF ------------------------------
    G_cvx, W_cvx = convex_nmf(XMat, 2, n_iter)
    exposures_cvx = XMat@G_cvx

    diagonals_cvx = W_cvx.sum(axis = 0)
    print(diagonals_cvx)
    exposures_cvx = exposures_cvx@np.diag(diagonals_cvx)
    signatures_cvx = np.diag(1/diagonals_cvx)@(W_cvx.T)

    # ------------------------- AE NMF -----------------------------------
    model_AE = AAUtoSig(feature_dim = 6, latent_dim = 2, relu_activation = [True, False])

    optimizer = torch.optim.Adam(params=model_AE.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    XMat_pd = pd.DataFrame(XMat)
    _,_,_,signatures_AE, exposures_AE,_ = train_AAUtoSig(epochs = n_iter, model = model_AE, x_train = XMat_pd, x_test = XMat_pd, criterion = criterion, optimizer = optimizer, batch_size = 8, do_plot = False, non_negative = "bases")
    diagonals_AE = signatures_AE.sum(axis = 0)
    print(diagonals_AE)
    exposures_AE = exposures_AE.T@np.diag(diagonals_AE)
    signatures_AE = np.diag(1/diagonals_AE)@(signatures_AE.T)
    print(i)
    # --------------------------- Plot results --------------------------------
    axs1[i,0].plot(list(range(total_n)), exposures[:,0], '-o', color = "forestgreen", label = "True")
    axs1[i,0].plot(list(range(total_n)), exposures[:,1], '-o', color = "lightgreen", label = "True")
    axs1[i,0].plot(list(range(total_n)), exposures_cvx[:,0], '-o', color = "blue", label = "Cvx NMF")
    axs1[i,0].plot(list(range(total_n)), exposures_cvx[:,1], '-o', color = "navy", label = "Cvx NMF")
    axs1[i,0].plot(list(range(total_n)), exposures_AE.iloc[:,0], '-o', color = "red", label = "AE")
    axs1[i,0].plot(list(range(total_n)), exposures_AE.iloc[:,1], '-o', color = "lightcoral", label = "AE")
    axs1[i,0].set_title('n = ' + str(total_n))

    axs1[i,1].plot(list(range(6)), signature1, '-o', color = "forestgreen", label = "True")
    axs1[i,1].plot(list(range(6)), signature2, '-o', color = "lightgreen", label = "True")
    axs1[i,1].plot(list(range(6)), signatures_cvx[0,:], '-o', color = "blue", label = "Cvx NMF")
    axs1[i,1].plot(list(range(6)), signatures_cvx[1,:], '-o', color = "navy" , label = "Cvx NMF")
    axs1[i,1].plot(list(range(6)), signatures_AE.iloc[0,:], '-o', color = "red", label = "AE")
    axs1[i,1].plot(list(range(6)), signatures_AE.iloc[1,:], '-o', color = "lightcoral",label = "AE")
    axs1[i,1].set_title('n = ' + str(total_n))
    if i == 0:
        axs1[i,0].set_title("Exposures \n n= " +  str(total_n))
        axs1[i,1].set_title("Signatures\n n= " +  str(total_n))
        axs1[i,1].legend()
plt.show()
'''

fig1, axs1 = plt.subplots(3,2, width_ratios = [4,1])
n = 50
# ---------------- Data Initialization ------------------------
signature1 = np.ones(6)/6
signature2 = np.array([0]*3 + [2]*3)/6
total_n = 3*n
exposures = np.array([18,2]*n + [10,10]*n + [2,18]*n).reshape(total_n,2)
XMat1 = np.random.poisson(18*signature1 + 2*signature2, size = (n,6))
XMat2 = np.random.poisson(10*signature1 + 10*signature2, size = (n,6))
XMat3 = np.random.poisson(2*signature1 + 18*signature2, size = (n,6))
XMat = np.concatenate((XMat1, XMat2, XMat3))
n_iter = 500



# -------------------------- NMF ---------------------------------
model_NMF = NMF(n_components=2, solver='mu', max_iter=n_iter, init="random")
exposures_NMF = model_NMF.fit_transform(XMat)
signatures_NMF = model_NMF.components_

diagonals_NMF = signatures_NMF.sum(axis = 1)
print(signatures_NMF.shape)
print(diagonals_NMF)
exposures_NMF = exposures_NMF@np.diag(diagonals_NMF)
signatures_NMF = np.diag(1/diagonals_NMF)@signatures_NMF

# -------------------------- CVX NMF ------------------------------
G_cvx, W_cvx = convex_nmf(XMat, 2, n_iter)
exposures_cvx = XMat@G_cvx

diagonals_cvx = W_cvx.sum(axis = 0)
print(diagonals_cvx)
exposures_cvx = exposures_cvx@np.diag(diagonals_cvx)
signatures_cvx = np.diag(1/diagonals_cvx)@(W_cvx.T)

# ------------------------- AE NMF -----------------------------------
model_AE = AAUtoSig(feature_dim = 6, latent_dim = 2, relu_activation = [True, False])

optimizer = torch.optim.Adam(params=model_AE.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

XMat_pd = pd.DataFrame(XMat)
_,_,_,signatures_AE, exposures_AE,_ = train_AAUtoSig(epochs = n_iter, model = model_AE, x_train = XMat_pd, x_test = XMat_pd, criterion = criterion, optimizer = optimizer, batch_size = 8, do_plot = False, non_negative = "bases")
diagonals_AE = signatures_AE.sum(axis = 0)
print(diagonals_AE)
exposures_AE = (exposures_AE.T@np.diag(diagonals_AE)).to_numpy()
signatures_AE = (np.diag(1/diagonals_AE)@(signatures_AE.T)).to_numpy()
names = ["NMF", "Cvx NMF", "AE"]
# --------------------------- Plot results --------------------------------
for i, est in enumerate(((exposures_NMF, signatures_NMF), (exposures_cvx, signatures_cvx), (exposures_AE, signatures_AE))):
    exp, sig = est
    axs1[i,0].plot(list(range(total_n)), exposures[:,0], '-o', color = "forestgreen", label = "True")
    axs1[i,0].plot(list(range(total_n)), exposures[:,1], '-o', color = "lightgreen", label = "True")
    axs1[i,0].plot(list(range(total_n)), exp[:,0], '-o', color = "blue", label = "estimated")
    axs1[i,0].plot(list(range(total_n)), exp[:,1], '-o', color = "navy", label = "estimated")
    axs1[i,0].set_title(names[i])

    axs1[i,1].plot(list(range(6)), signature1, '-o', color = "forestgreen", label = "True")
    axs1[i,1].plot(list(range(6)), signature2, '-o', color = "lightgreen", label = "True")
    axs1[i,1].plot(list(range(6)), sig[0,:], '-o', color = "blue", label = "estimated")
    axs1[i,1].plot(list(range(6)), sig[1,:], '-o', color = "navy" , label = "estimated")
    axs1[i,1].set_title(names[i])
    if i == 0:
        axs1[i,0].set_title("Exposures \n" +  names[i])
        axs1[i,1].set_title("Signatures\n" + names[i])
        axs1[i,1].legend()


plt.show()
