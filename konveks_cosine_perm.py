from xmlrpc.client import boolean
import cvxpy as cp
import numpy as np

# Square cosine matrix A
n = 3

true = np.random.rand(n, 96)
est = np.random.rand(n, 96)
import scipy.spatial as sp

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

print("Original value", A.trace())
print("Optimal value", problem.solve())

print("Original var")
print(A)
print("Optimal var")
print(Y.value) # A numpy ndarray.

print(P.value)
print(P.value.argmax(axis = 1))