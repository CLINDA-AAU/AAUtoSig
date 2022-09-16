from xmlrpc.client import boolean
import cvxpy as cp
import numpy as np
from functions import cosine_perm
from scipy.optimize import linear_sum_assignment
import scipy.spatial as sp

n = 4

true = np.abs(np.random.rand(n, 96))
est = np.abs(np.random.rand(n, 96))
import scipy.spatial as sp

A = sim = 1 - sp.distance.cdist(est, true, 'cosine')

#hungarian algorithm




def test_cvx(n):
    # Square cosine matrix A
    n = n

    true = np.abs(np.random.rand(n, 96))
    est = np.abs(np.random.rand(n, 96))
    A = sim = 1 - sp.distance.cdist(est, true, 'cosine')

    # Constraint variables
    Y = cp.Variable((n,n))
    P = cp.Variable((n,n), boolean = True)

    #Define problem
    problem = cp.Problem(cp.Maximize(cp.trace(Y)), 
                            [Y == P@A, #rowwise permutaiton, colwise: Y == A@P
                            cp.sum(P, axis = 1) == 1,
                            cp.sum(P, axis = 0) == 1])

    problem.solve()

    sim_cvx = Y.value # A numpy ndarray.
    perm_cvx = P.value.argmax(axis = 1)
    row_ind, col_ind  = linear_sum_assignment(-A.T)

    #sim_pe, perm_pr = cosine_perm(est, true)
    #return(sim_cvx, perm_cvx)
    #return(np.mean(sim_cvx.diagonal()))

    print(set(A[perm_cvx].diagonal()) == set(A[row_ind, col_ind]) ) 
    print(row_ind, col_ind)
    print(perm_cvx)
    print(A)
    print((A.T[:,col_ind]).T)
    print(A[perm_cvx])

        #return(A, A[perm_cvx], A[col_ind])

"""

print(A)
row_ind, col_ind  = linear_sum_assignment(-A)
print(row_ind, col_ind)
sim, pe = test_cvx(A)
print(pe)
print(sim)
print(A[row_ind, col_ind].sum())
print(A[pe].diagonal().sum())
"""
asd = [test_cvx(4) for _ in range(1)]
print(asd)







"""

print("Original value", A.trace())
print("Optimal value", problem.solve())

print("Original var")
print(A)
print("Optimal var")
"""