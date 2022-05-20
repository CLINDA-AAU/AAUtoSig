# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from functions import simulate_counts, plotsigs, cosine_perm, simulate_mixedLittle
# This metod initializes signature and exposure matrices. A is the signatures and S is the exposures
# Do you need to think about scaling of the matrices? yes - at some point

def deepNMF(layer_sizes, X_mat):
    def initailize_matrices(X, nsig):
        #hvad er der med de her NaN'er?
        #A_l,_,_ = vca(X, nsig, verbose = False)
        # exposures were supposed to be estimated by FCLS, by i do this by NMF's 
        # nnls since I do not wish to constrain the exposures to sum to one
        nmf = NMF(nsig)
        S_l = nmf.fit_transform(X.T)
        A_l = nmf.components_
        #S_l = nmf.fit_transform(X.T, H = A_l)
        return A_l.T, S_l.T
    
    layers = len(layer_sizes)
    nsig = layer_sizes[-1]

    #pretraining
    X_np = X_mat.to_numpy()
    S_list = [X_np]
    A_list = [] 
    for l in range(layers):
        X = S_list[l]
        A, S = initailize_matrices(X, layer_sizes[l])
        #maybe introduce another stopping criterion

        for _ in range(50):
            #A_bar = np.stack((A, A_vec))
            #X_bar = np.stack((X, X_vec))
            A = A*(X@S.T)/np.clip(((A@S)@S.T), a_min = 1e-3, a_max = np.inf)
            S = S*(A.T@X)/np.clip(((A.T@A)@S), a_min = 1e-3, a_max = np.inf)
        A_list.append(A)
        S_list.append(S)
    A_list.append(np.eye(nsig))
    
    #fine_tuning 
    for _ in range(50):
        for l in range(layers):
            
            
            psi_lm1 = np.eye(96)
            S_tildel = np.eye(layer_sizes[l])
            if l == 0:
                for mat in A_list[1::]:S_tildel = S_tildel@mat
            else:
                #this slicer does not indclude the last idx
                for mat in A_list[0:l]: psi_lm1 = psi_lm1@mat
                for mat in A_list[-(layers-l):]: S_tildel = S_tildel@mat
            S_tildel = S_tildel @ S_list[-1]

            A_list[l] = A_list[l]*(psi_lm1.T@X_np@S_tildel.T)/np.clip((psi_lm1.T@psi_lm1@A_list[l]@S_tildel@S_tildel.T), a_min = 1e-3, a_max = np.inf)
            S_list[l+1] = S_list[l+1]*((psi_lm1@A_list[l]).T@X_np)/np.clip(((psi_lm1@A_list[l]).T@psi_lm1@A_list[l]@S_list[l+1]), a_min = 1e-3, a_max = np.inf)
     
    sigs_gg = pd.DataFrame(A_list[0]@A_list[1]@A_list[2])
    return(sigs_gg, S_list[-1])

X, true_sigs = simulate_mixedLittle(7, 2000)
trinucleotide = X.index
mutation = [t[2:5] for t in trinucleotide]
sigs_gg, _ = deepNMF([13, 10, 7], X)
perm = cosine_perm(sigs_gg.T,true_sigs.T)
sigs_gg= sigs_gg[perm[1]]
plotsigs(trinucleotide, mutation, sigs_gg.to_numpy(), 7, "wuhu?")
plotsigs(trinucleotide, mutation, true_sigs.to_numpy(), 7, "True signatures")