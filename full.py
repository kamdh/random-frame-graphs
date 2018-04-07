from frame import Frame
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh, eigvals, eig, eigh
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from spectrum import compute_spectrum, non_backtracking_matrix
 
epsilon = 0.001
offset = 0.5
N = 400
num_eig = 10
d = 6
ps = {0: 0.5, 1: 0.5}
eset = [(0,0), (0,1), (1,0), (1,1)]
deg = {(0,0): d+9, (0,1): d, (1,0): d, (1,1): d+9}

g = Frame(nx.Graph(eset), p = ps, deg = deg)
gsamp = g.sample(N)
# rho = compute_spectrum(gsamp)
# rho.sort()

Pdiag, K, Q, P = g.base_matrices()
As = nx.adjacency_matrix(gsamp).todense()
I = nx.incidence_matrix(gsamp, oriented=True).todense()

D,V = eigh(K)
print "frame eigenvalues:"
print D
Ds,Vs = eigh(As)
Ds = np.flip(Ds, axis=0)
Vs = np.flip(Vs, axis=1)
print "ESD leading eigenvalues:"
print Ds[:num_eig]
Ps = As/np.tile(np.sum(As,axis=1), (1, As.shape[1])).astype(float)
DPs, VPs = eig(Ps)
Bs = non_backtracking_matrix(gsamp)
Bs = sp.csr_matrix(Bs)
DBs, VBs = eigs(Bs, k=num_eig, maxiter=Bs.shape[0]*30)
# DBs = np.flip(DBs, axis=0)
# VBs = np.flip(VBs, axis=1)
print "B leading eigenvalues:"
print DBs
T = np.hstack((I,-I)).dot(VBs)

plt.ion()
fig = plt.figure()

ax = fig.add_subplot(311)
ax.imshow(Vs[:,:num_eig].real)
ax.set_aspect('auto')

ax = fig.add_subplot(312)
ax.imshow(VBs.real)
ax.set_aspect('auto')

ax = fig.add_subplot(313)
ax.imshow(T.real)
ax.set_aspect('auto')


