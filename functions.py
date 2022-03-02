import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as sp
from random import sample
from itertools import permutations

def simulate_counts(nsigs, npatients, loglinear = False):
  #Arrange COSMIC to be the same ordering as count data
  #COSMIC = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\COSMIC_SIGNATURES\COSMIC_v3.2_SBS_GRCh37.txt', sep='\t', index_col=0)
  COSMIC = pd.read_csv(r'COSMIC\COSMIC_v3.2_SBS_GRCh37.txt', sep = '\t', index_col=0)
  context = COSMIC.index
  mutation = [s[2:5] for s in context]
  COSMIC['mutation'] = mutation
  COSMIC = COSMIC.sort_values('mutation')
  mutation = COSMIC['mutation']
  context = COSMIC.index
  COSMIC = COSMIC.drop('mutation', axis = 1)

  patients = ['Patient' + str(i) for i in range(1,(npatients+1))]

  sig_names = sample(list(COSMIC.columns), nsigs)
  sigs_true = COSMIC[sig_names]
  sigs = sigs_true[:]
  if loglinear:
      sigs[sigs == 0] = 1e-7
      # Der er 65 nuller i COSMIC i alt
      sigs = np.log(sigs)

  def generate_exposure(nsigs):
    zinf = np.random.binomial(n = 1, p = 0.09, size = nsigs)>0 
    not_zinf = [not z for z in zinf]
    #parametrized negative binomial with mean 600
    total_muts = np.random.negative_binomial(p =1- 300/301, n = 2, size = 1)
    distribution = np.random.dirichlet(alpha=[1]*nsigs, size= 1)

    res = (np.multiply(not_zinf, distribution)*total_muts).tolist()
    #because it somehow made a list of lists
    return(res[0])
  E = [generate_exposure(nsigs) for _ in range(npatients)]
  Exposures = pd.DataFrame(E).transpose()

  Exposures.columns = patients
  Exposures.index = sig_names
  
  V = pd.DataFrame(np.round(np.dot(sigs, Exposures),0))
  V = np.exp(V) if loglinear else V
  V.columns = patients
  V.index = context

  return((V, sigs_true, Exposures))

a,b,c = simulate_counts(4, 15, loglinear=True)
print(a)

def plotsigs(context, mutation, signatures, nsigs, title):
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
        if i == 0:
            plt.title(title)
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
            
    return((best_pe, list(best_idx)))
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