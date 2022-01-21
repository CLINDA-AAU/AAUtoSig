import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as sp
from itertools import permutations


def plotsigs(context, mutation, signatures, nsigs):
    colors = {'C>A': 'r', 'C>G': 'b', 'C>T': 'g', 
                'T>A' : 'y', 'T>C': 'c','T>G' : 'm' }
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    for i in range(nsigs):
        plt.subplot(nsigs,1, (i+1))
        #plt.figure(figsize=(20,7))
        plt.bar(x = context, 
                height =  signatures[:,i]/np.sum(signatures[:,i]), 
                color = [colors[i] for i in mutation])
        plt.xticks([])
    #plt.legend(handles,labels)
    #plt.xticks(rotation=90)
    plt.show()
    


#This function takes to matrices of equal size, calculates the row-wise cosine
#similarity between the two matrices and permutes the rows to have the highest
#possible diagonal values. 
#It returns the permuted cosine matrix, and the arrangement of matrix B's rows
#that generates the highest cosine similarity with the corresponding rows in A
#A = np.random.rand(3,2)
#B = np.random.rand(3,2)
def cosine_perm(A,B):
    def acc(x):
        return(np.sum(np.diag(x)))
    #This operation creates the cosine distance matrix between rows in A and rows 
    #in B, where the rows in sim represent the rows in A and the columns in sim 
    #represent the rows in B.
    sim = 1 - sp.distance.cdist(A, B, 'cosine')
    curr_x = acc(sim)
    best_pe = sim
    best_idx =range(A.shape[0])
    for pe, idx in zip(permutations(sim), permutations(best_idx)):
        pe = np.array(pe)
        if(acc(pe)>curr_x):
            curr_x = acc(pe)
            best_pe = pe
            best_idx = idx
            
    return((best_pe, best_idx))
'''
res = np.round(cosine_perm(A, B)[0],2)


signatures = ["S1", "S2", "S3"]



fig, ax = plt.subplots()
im = ax.imshow(res)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(signatures)))
ax.set_xticklabels(signatures)
ax.set_yticks(np.arange(len(signatures)))
ax.set_yticklabels(signatures)


# Loop over data dimensions and create text annotations.
for i in range(len(signatures)):
    for j in range(len(signatures)):
        text = ax.text(j, i, res[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()
'''