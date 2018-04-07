from frame import Frame
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh, eigvals, eig, eigh
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from spectrum import compute_spectrum, non_backtracking_matrix
from sklearn.cluster import KMeans

def compute_c(K):
    R = K/np.tile(np.sum(K, axis=1), (1, K.shape[1])).astype(float)
    c = 0.
    ctest = 0.
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if K[i,j] > 0:
                ctest = np.sqrt( (K[i,j] - 1.)/(K[i,j] * K[j,i]) + \
                                 (K[j,i] - 1.)/(K[i,j] * K[j,i]) )
            if ctest > c:
                c = ctest
    return c

epsilon = 0.001
offset = 0.5
scale = 1
N = 400
num_eig = 10
# ps = {0: 0.05, 1: 0.25, 2: 0.4, 3: 0.25, 4: 0.05}
# deg = {(0,1): 10, (1,2):8, (2,3):5, (3,4): 2,
#        (1,0): 2, (2,1):5, (3,2):8, (4,3):10}
# ps = {0: 0.125, 1: 0.125, 2: 0.75}
# deg = {(0,1): 3, (1,2):12, (1,0): 3, (2,1):2}
ps = {0: 0.2, 1: 0.8}
deg = {(0,1): 20, (1,0): 5}

if __name__=="__main__":
    # # bad
    # deg = {(0,1): 3, (1,2):6, (2,3):2, 
    #        (1,0): 2, (2,1):6, (3,2): 10}
    # good
    # deg = {(0,1): 50, (1,2):40, (2,3):25, (3,4):10,
    #        (1,0): 10, (2,1):25, (3,2):40, (4,3):50}

    if scale is not None:
        for key, val in deg.iteritems():
            deg[key] = val * scale
    g = Frame(nx.DiGraph(nx.path_graph(len(ps))), p = ps, deg = deg)
    # xs = np.linspace(-2,2,701)
    # Aspec = g.spectrum(xs, epsilon=epsilon, offset=offset)
    # plt.ion()
    # plt.figure()
    # plt.plot(xs, np.abs(Aspec), 'b-', linewidth=2)
    # raw_input()
    gsamp = g.sample(N)

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
