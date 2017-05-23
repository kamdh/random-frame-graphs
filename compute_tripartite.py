#!/usr/bin/env python
from spectrum import compute_spectrum
import tripartite as tri
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh,eigvals
from frame import Frame
import networkx as nx
#from IPython import embed
'''
Generate a random regular tripartite graph.
Then, compute its eigenspectrum and plot the empirical spectral density.
'''
#def main():
n=72
laplacian = True
if laplacian:
    xs = np.linspace(-20,0,801)
    xlim=(0,20)
    alpha=1
else:
    alpha=0
    xs=np.linspace(-6,7,801)
    xlim=(-6,8)
simple = True
epsilon =0.001
graph_file="../figures/tri_pA_0.125_pB_0.125_kAB_3_kAC_6_kBC_12_n_360.gml"
# pA = 0.125
# pB = 0.25
# kAB = 2
# kAC = 10
# kBC = 5
pA = 0.125
pB = 0.125
kAB = 3
kAC = 6
kBC = 12
# pA = 0.125
# pB = 0.25
# kAB = 2
# kAC = 0
# kBC = 5
# pA=1/3.0
# pB=1/3.0
# kAB=3
# kAC=3
# kBC=3
## derived parameters:
pC = 1.-pA-pB
kBA = pA/pB*kAB
kCB = pB/pC*kBC
kCA = pA/pC*kAC
print "Generating graphs"
g = tri.tripartite_regular_configuration_graph(n, pA, pB, kAB, kAC, kBC,
                                               verbose=True) 
print "Diagonalizing matrix"
#embed()    
L = compute_spectrum(g, laplacian=laplacian)
print "Plotting"
plt.ion()
plt.figure(0)
plt.hist(L, bins=121, normed=True)
print "Computing spectrum analytically"
print "  1. tripartite specific code"
Lanalytic = tri.tripartite_regular_spectrum(xs, pA, pB, kAB, kAC, kBC,
                                            epsilon=epsilon, alpha=alpha)
print "  2. general Frame code"
triFrame = Frame(nx.complete_graph(3, create_using=nx.DiGraph()),
                 p={0:pA, 1:pB, 2:pC},
                 deg={(0,1):kAB, (0,2):kAC, (1,0):kBA, (1,2):kBC,
                      (2,0):kCA, (2,1):kCB})
Lanalytic2 = triFrame.spectrum(xs, epsilon=epsilon, alpha=alpha)
print "  3. base graph"
if laplacian:
    pass
else:
    Lbase3=nx.linalg.spectrum.adjacency_spectrum(triFrame,weight='deg')
# reflect eigs about origin for laplacians
if laplacian:
    xs = -xs
plt.hold(True)
#plt.plot(xs, np.abs(Lanalytic), 'g-', linewidth=2)
plt.plot(xs, np.abs(Lanalytic2), 'r-', linewidth=2)
plt.ylim((0, 0.5))
plt.xlim(xlim)
# A = tri.block_gaussian_guess(n, pA, pB)
# Lguess = eigvalsh(A, check_finite=False)
# Lguess /= np.sqrt(n)
# plt.hist(Lguess, bins=301, normed=True)
#from IPython import embed
if not laplacian:
    P,K,Q=triFrame.base_matrices()
    Keigs=eigvals(K)
    Kmax=Keigs.max()
    plt.plot((Kmax,Kmax),(0.01,0.5),'g--')

raw_input("Enter to continue")

#embed()
nx.write_gml(g, graph_file)


# from nonbacktracking import *
# B=nonbacktracking_matrix(g)
# from scipy.sparse.linalg import eigs as speigs
# B_eig1=speigs(B,k=1)[0][0]
# print B_eig1
# Bsm=nonbacktracking_matrix_guess(triFrame)
# Bsm_eigs=eigvals(Bsm.todense())
# print Bsm_eigs.max()
# B_eigs=eigvals(B.todense())
# plt.figure()
# plt.scatter(B_eigs.real,B_eigs.imag,c='k', marker='.')
# plt.hold(True)
# plt.scatter(Bsm_eigs.real,Bsm_eigs.imag,edgecolors='b',
#            marker='o',facecolors='none')

# if __name__ == '__main__':
#    main()
