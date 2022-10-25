import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as sp
from random import sample
from itertools import permutations
import sklearn.metrics 


# Pei et al. uses data on the pentanucleotide from to extract the mutational signatures. There
# is however, no readily avaliable pentanucleotide mutational signatures in COSMIC.
# This method 'expands' trinucelotide signatures to pentanucleotide signatures by a 16-fold expan-
# sion of each datapoint. 
# Dilemma: We can either expand them unifromly and infer no extra information but a lot of extra
# parameters in the model, which could be unfarvorable for the model, or the we can expand using 
# something like a dirichlet and and the expansion may then not reflect something biologically
# accurate. This may also be unfarvorable for the model, or it may not matter.
# This model takes a trinucleotide SBS mutational signature, and expands it to a pentanucleotide 
# from.

def expand_SBS(sig):
  #context = sig.index
  #bases = ['A', 'C', 'G', 'T']
  #penta = [ v + c + h for c in context for v in bases for h in bases]
  def expand(val):
    val_exp = val * np.random.dirichlet(alpha = [1]*(16), size = 1)
    return(val_exp)
  penta_SBS = [expand(val)[0] for val in sig]
  res = [item for sublist in penta_SBS for item in sublist] #used to be a df
  #res.axis = penta
  return res



def simulate_counts(nsigs, npatients, pentanucelotide = False, sig_names = None):
  #Arrange COSMIC to be the same ordering as count data
  COSMIC = pd.read_csv("COSMIC/COSMIC_v3.2_SBS_GRCh37.txt", sep = '\t', index_col=0)
  context = COSMIC.index
  mutation = [s[2:5] for s in context]
  COSMIC['mutation'] = mutation
  COSMIC = COSMIC.sort_values('mutation')
  mutation = COSMIC['mutation']
  context = COSMIC.index
  COSMIC = COSMIC.drop('mutation', axis = 1)

  patients = ['Patient' + str(i) for i in range(1,(npatients+1))]

  if not sig_names: sig_names = sample(list(COSMIC.columns), nsigs)
  sigs = COSMIC[sig_names]
  if pentanucelotide:
    sigs = pd.DataFrame([expand_SBS(sigs.iloc[:,i]) for i in range(nsigs)]).T
    bases = ['A', 'C', 'G', 'T']
    penta = [ v + c + h for c in context for v in bases for h in bases]
  def generate_exposure(nsigs):
    #zinf = np.random.binomial(n = 1, p = 0.09, size = nsigs + 1)>0 
    #not_zinf = [not z for z in zinf]
    not_zinf = np.random.binomial(n = 1, p = 0.09, size = nsigs) == 0 
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

  sigs.columns = sig_names
  sigs.index = context if (not pentanucelotide) else penta

  V = pd.DataFrame(np.round(np.dot(sigs, Exposures),0))
  V.columns = patients
  V.index = context if (not pentanucelotide) else penta

  return((V, sigs, Exposures))


def simulate_mixedLittle(nsigs, npatients, pentanucleotide = False):
  #Arrange COSMIC to be the same ordering as count data
  COSMIC = pd.read_csv(r'COSMIC\COSMIC_v3.2_SBS_GRCh37.txt', sep = '\t', index_col=0)
  context = COSMIC.index
  mutation = [s[2:5] for s in context]
  COSMIC['mutation'] = mutation
  COSMIC = COSMIC.sort_values('mutation')
  mutation = COSMIC['mutation']
  context = COSMIC.index
  COSMIC = COSMIC.drop('mutation', axis = 1)

  patients = ['Patient' + str(i) for i in range(1,(npatients + 1))]

  sig_names = sample(list(COSMIC.columns), nsigs)
  sigs = COSMIC[sig_names]

  mix_idx = sample(range(nsigs), 2)
  mix_sig = np.log(3*(COSMIC.iloc[:,mix_idx].dot([1, 1])) + 1) #a steep log that has root in zero
  sigs = np.concatenate((sigs, mix_sig.to_numpy().reshape((96, 1))), axis = 1)

  if pentanucleotide:
    sigs = pd.DataFrame([expand_SBS(sigs[:,i]) for i in range(nsigs + 1)]).T
    bases = ['A', 'C', 'G', 'T']
    penta = [ v + c + h for c in context for v in bases for h in bases]

  def generate_exp(nsigs):
    #zinf = np.random.binomial(n = 1, p = 0.09, size = nsigs + 1)>0 
    #not_zinf = [not z for z in zinf]
    not_zinf = np.random.binomial(n = 1, p = 0.09, size = nsigs + 1) == 0 
    # parametrized negative binomial with mean 600. This is not gonna be far from the mean of the total counts
    # as there is two points where some exposures are set to 0
    total_muts = np.random.negative_binomial(p = 1 - 400/401, n = 2, size = 1)
    distribution = np.random.dirichlet(alpha = [1]*(nsigs + 1), size = 1)
    
    #because it somehow made a list of lists
    exp = (np.multiply(not_zinf, distribution)*total_muts).tolist()[0]

    # den transformerede signatur er, hvis de begge to er aktive i genomet. Hvis en af de udvalgte ikke
    # er til stede i genomet får man ikke den ikke-lineære effekt. Så får man bare den lineære effekt af
    # at have den, der er til stede.
    # mix_sig er på den nsigs plads i signaturmatricen idet der nu er nsigs + 1 signaturer, men python 
    # indekserer fra 0. 
    if any([exp[i] == 0 for i in mix_idx]): 
      exp[nsigs] = 0
    # hvis begge signaturer fra der indgår i ikke-lineariteten er til stede sættes deres exposure til 0
    # da kommer de kun til udtryk i mixet
    else:
      exp[mix_idx[0]] = 0
      exp[mix_idx[1]] = 0

    return(exp)

  E = pd.DataFrame([generate_exp(nsigs) for _ in range(npatients)]).T

  V = pd.DataFrame(np.round(np.dot(sigs, E),0))
  V.columns = patients
  V.index = penta if pentanucleotide else context

  sigs = pd.DataFrame(sigs).iloc[:, :-1]
  sigs.columns = sig_names
  sigs.index = penta if pentanucleotide else context

  return((V, sigs))



def simulate_mixedBIG(nsigs, npatients):
  #Arrange COSMIC to be the same ordering as count data
  COSMIC = pd.read_csv(r'COSMIC\COSMIC_v3.2_SBS_GRCh37.txt', sep = '\t', index_col=0)
  context = COSMIC.index
  mutation = [s[2:5] for s in context]
  COSMIC['mutation'] = mutation
  COSMIC = COSMIC.sort_values('mutation')
  mutation = COSMIC['mutation']
  context = COSMIC.index
  COSMIC = COSMIC.drop('mutation', axis = 1)

  patients = ['Patient' + str(i) for i in range(1,(npatients + 1))]

  sig_names = sample(list(COSMIC.columns), nsigs)
  sigs = COSMIC[sig_names]

  mix_idx = sample(range(nsigs), 4)
  mix_sig1 = np.log(3*(COSMIC.iloc[:,mix_idx[0:2]].dot([1, 1])) + 1) #a steep log that has root in zero
  mix_sig2 = np.log(4*(COSMIC.iloc[:,mix_idx[2:4]].dot([1, 1])) + 1) #a steep log that has root in zero
  
  sigs = np.concatenate((sigs, mix_sig1.to_numpy().reshape((96, 1)), mix_sig2.to_numpy().reshape((96, 1))), axis = 1)

  def generate_exp(nsigs):
    not_zinf = np.random.binomial(n = 1, p = 0.09, size = nsigs + 2) == 0 
    # parametrized negative binomial with mean 600. This is not gonna be far from the mean of the total counts
    # as there is two points where some exposures are set to 0
    total_muts = np.random.negative_binomial(p = 1 - 500/501, n = 2, size = 1)
    distribution = np.random.dirichlet(alpha = [1]*(nsigs + 2), size = 1)
    
    #because it somehow made a list of lists
    exp = (np.multiply(not_zinf, distribution)*total_muts).tolist()[0]

    # de transformerede signaturer er aktive, hvis begge delkoponenter to er aktive i genomet. Hvis en 
    # af de udvalgte ikke er til stede i genomet får man ikke den ikke-lineære effekt. Så får man bare 
    # den lineære effekt af at have den, der er til stede.
    # mix_sig1 og mix_sig2 er på den nsigs og nsig + 1 plads i signaturmatricen idet der nu er nsigs + 1 
    # signaturer, men python  indekserer fra 0. 
    if any([exp[i] == 0 for i in mix_idx[0:1]]): 
      exp[nsigs] = 0
    # hvis begge signaturer fra der indgår i ikke-lineariteten er til stede sættes deres exposure til 0
    # da kommer de kun til udtryk i mixet
    else:
      exp[mix_idx[0]] = 0
      exp[mix_idx[1]] = 0


    if any([exp[i] == 0 for i in mix_idx[2:3]]): 
      exp[nsigs] = 0

    else:
      exp[mix_idx[2]] = 0
      exp[mix_idx[3]] = 0
    return(exp)

  E = pd.DataFrame([generate_exp(nsigs) for _ in range(npatients)]).T

  V = pd.DataFrame(np.round(np.dot(sigs, E),0))
  V.columns = patients
  V.index = context

  sigs = pd.DataFrame(sigs).iloc[:, :-2]
  sigs.columns = sig_names
  sigs.index = context

  return((V, sigs))



def plotsigs(context, mutation, signatures, nsigs, title):
    colors = {'C>A': 'r', 'C>G': 'b', 'C>T': 'g', 
                'T>A' : 'y', 'T>C': 'c','T>G' : 'm' }
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    for i in range(nsigs): signatures[:,i] = signatures[:,i]/np.sum(signatures[:,i])
    max_val = signatures.max().max()
    for i in range(nsigs):
        plt.subplot(nsigs,1, (i+1))
        #plt.figure(figsize=(20,7))
        plt.bar(x = context, 
                height =  signatures[:,i], 
                color = [colors[i] for i in mutation])
        plt.xticks([])
        plt.ylim( [ 0, max_val ] ) 
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
    best_idx = range(A.shape[0])
    for pe, idx in zip(permutations(sim), permutations(best_idx)):
        pe = np.array(pe)
        if(acc(pe)>curr_x):
            curr_x = acc(pe)
            best_pe = pe
            best_idx = idx
            
    return((best_pe, list(best_idx)))

from xmlrpc.client import boolean
import cvxpy as cp
import numpy as np

def cosine_cvx(est_set, ref_set):
    #This operation creates the cosine distance matrix between rows in A and rows 
    #in B, where the rows in sim represent the rows in A and the columns in sim 
    #represent the rows in B.
    sim = 1 - sp.distance.cdist(est_set, ref_set, 'cosine')
    nsigs = est_set.shape[0]
    Y = cp.Variable((nsigs,nsigs))
    P = cp.Variable((nsigs,nsigs), boolean = True)

#Define problem
    problem = cp.Problem(cp.Maximize(cp.trace(Y)), 
                        [Y == P@sim, #rowwise permutaiton, colwise: Y == A@P
                        cp.sum(P, axis = 1) == 1,
                        cp.sum(P, axis = 0) == 1])


    problem.solve()
    perm = P.value.argmax(axis = 1)
    return((Y.value, perm))

from scipy.optimize import linear_sum_assignment


# optimal signature matching using the Hungarian algorithm
def cosine_HA(est_set, ref_set):
    #This operation creates the cosine distance matrix between rows in A and rows 
    #in B, where the rows in sim represent the rows in A and the columns in sim 
    #represent the rows in B.
    sim = 1 - sp.distance.cdist(est_set, ref_set, 'cosine')
    _, col_ind  = linear_sum_assignment(-sim.T)
    return((sim.T[:,col_ind]).T, col_ind)

#A = np.random.rand(3,2)
#B = np.random.rand(3,2)
'''
result = cosine_cvx(A, B)
res = np.round(result[0],2)

print(result)

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
