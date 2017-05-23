import networkx as nx
import numpy as np
#from IPython import embed

def block_gaussian_guess(n, pA, pB):
    '''
    Generate randn matrices X, Y, Z
    Return:

    A = [[ 0  X  Y ]
         [ X' 0  Z ]
         [ Y' Z' 0 ]]
    '''
    nA = int(np.round(n * pA))
    nB = int(np.round(n * pB))
    nC = n - nA - nB
    X = np.random.randn(nA,nB)
    Y = np.random.randn(nA,nC)
    Z = np.random.randn(nB,nC)
    A = np.bmat([[np.zeros((nA,nA)), X, Y],
                 [np.transpose(X), np.zeros((nB,nB)), Z],
                 [np.transpose(Y), np.transpose(Z), np.zeros((nC,nC))]])
    return A

def tripartite_regular_spectrum(xs, pA, pB, kAB, kAC, kBC, 
                                epsilon=0.01, alpha=0):
    '''
    Compute a tripartite regular random graph's spectrum.

    Parameters
    ==========
    xs, numpy array
      Points at which to compute the spectrum.
    pA, float, 0 <= pA <= 1
      Size of set A.
    pB, float, 0 <= pB <= 1
      Size of set B.
    kAB, float, kAB >= 0
      Degree from set A to B.
    kAC, float, kAC >= 0
      Degree from set A to C.
    kBC, float, kBC >= 0
      Degree from set B to C.
    epsilon, float, optional
      Distance from real axis to perform inversion.
    alpha, {0,1}, optional
      Set to 0 for adjacency spectrum, 1 for graph laplacian.
    '''
    pC = 1-pA-pB
    kBA = pA/pB*kAB
    kCB = pB/pC*kBC
    kCA = pA/pC*kAC
    ## method 1, fixed point finding
    # from scipy.optimize import fixed_point
    # def _yfun_iter(y, z, alpha, kAB, kAC, kBA, kBC, kCA, kCB):
    #     (yAB,yAC,yBA,yBC,yCA,yCB) = y
    #     return np.array( [
    #         -1./(z + alpha*(kAB + kAC) + (kAB - 1)*yBA + kAC*yCA),
    #         -1./(z + alpha*(kAB + kAC) + kAB*yBA + (kAC - 1)*yCA),
    #         -1./(z + alpha*(kBA + kBC) + (kBA - 1)*yAB + kBC*yCB),
    #         -1./(z + alpha*(kBA + kBC) + kBA*yAB + (kBC - 1)*yCB),
    #         -1./(z + alpha*(kCB + kCA) + kCB*yBC + (kCA - 1)*yAC),
    #         -1./(z + alpha*(kCB + kCA) + (kCB - 1)*yBC + kCA*yAC) ])
    # y0 = [0+1.j, 0+1.j, 0+1.j, 0+1.j, 0+1.j, 0+1.j]
    # y_soln = lambda z: fixed_point(_yfun_iter, y0, args=(z, alpha, kAB, kAC, 
    #                                                      kBA, kBC, kCA, kCB))
    ## method 2, root finding
    from scipy.optimize import root
    def _yfun_root(y, z, alpha, kAB, kAC, kBA, kBC, kCA, kCB):
        yAB,yAC,yBA,yBC,yCA,yCB = y
        return np.array( [
            yAB*(z + alpha*(kAB + kAC) + (kAB - 1)*yBA + kAC*yCA)+1,
            yAC*(z + alpha*(kAB + kAC) + kAB*yBA + (kAC - 1)*yCA)+1,
            yBA*(z + alpha*(kBA + kBC) + (kBA - 1)*yAB + kBC*yCB)+1,
            yBC*(z + alpha*(kBA + kBC) + kBA*yAB + (kBC - 1)*yCB)+1,
            yCA*(z + alpha*(kCB + kCA) + kCB*yBC + (kCA - 1)*yAC)+1,
            yCB*(z + alpha*(kCB + kCA) + (kCB - 1)*yBC + kCA*yAC)+1 ])
    def _yfun_root_real(y, z, alpha, kAB, kAC, kBA, kBC, kCA, kCB):
        yABr,yABi,yACr,yACi,yBAr,yBAi,yBCr,yBCi,yCAr,yCAi,yCBr,yCBi = y
        y_eval = _yfun_root([complex(yABr,yABi), complex(yACr,yACi),
                             complex(yBAr,yBAi), complex(yBCr,yBCi),
                             complex(yCAr,yCAi), complex(yCBr,yCBi)],
                            z, alpha, kAB, kAC, kBA, kBC, kCA, kCB)
        yAB,yAC,yBA,yBC,yCA,yCB = y_eval
        return [yAB.real, yAB.imag, yAC.real, yAC.imag, yBA.real, yBA.imag,
                yBC.real, yBC.imag, yCA.real, yCA.imag, yCB.real, yCB.imag]
    y0 = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]
    y_soln_real = lambda z: root(_yfun_root_real, y0, 
                                 args=(z, alpha, kAB, kAC, 
                                       kBA, kBC, kCA, kCB))
    def y_soln(z):
        sol = y_soln_real(z)
        ypt = sol.x
        return np.array( [complex(ypt[0],ypt[1]), 
                          complex(ypt[2],ypt[3]),
                          complex(ypt[4],ypt[5]),
                          complex(ypt[6],ypt[7]),
                          complex(ypt[8],ypt[9]),
                          complex(ypt[10],ypt[11])] )
    def _xfun(y, z, alpha, kAB, kAC, kBA, kBC, kCA, kCB):
        (yAB,yAC,yBA,yBC,yCA,yCB) = y
        return np.array( [
            -1./(z + alpha*(kAB + kAC) + kAB*yBA + kAC*yCA),
            -1./(z + alpha*(kBA + kBC) + kBA*yAB + kBC*yCB),
            -1./(z + alpha*(kCB + kCA) + kCA*yAC + kCB*yBC) ])
    # compute the spectrum, setup helper functions
    x_soln = lambda z: _xfun(y_soln(z), z, alpha, kAB, kAC, kBA, kBC, kCA, kCB)
    density = lambda x: np.imag(
        np.dot([pA,pB,pC], x_soln(x + epsilon*1.0j)) ) / np.pi
    density_vec = np.vectorize(density, otypes=[np.float])
    return density_vec(xs)

# def tripartite_regular_graph(n, pA, pB, kAB, kAC, kBC):
#     '''
#     Generate a tripartite regular random graph.

#     Parameters
#     ==========
#     n, int
#       Number of nodes.
#     pA, float, 0 <= pA <= 1
#       Size of set A.
#     pB, float, 0 <= pB <= 1
#       Size of set B.
#     kAB, float, kAB >= 0
#       Degree from set A to B.
#     kAC, float, kAC >= 0
#       Degree from set A to C.
#     kBC, float, kBC >= 0
#       Degree from set B to C.

#     The graph is generated by sampling k out-regular blocks X, Y, Z
#     for the corresponding k and setting the adjacency matrix 
#     equal to:

#     A = [[ 0  X  Y ]
#          [ X' 0  Z ]
#          [ Y' Z' 0 ]]
#     '''
#     nA = int(np.round(n * pA))
#     nB = int(np.round(n * pB))
#     nC = n - nA - nB
#     def _sample_block(n1, n2, k12):
#         # S1 -> S2 connections
#         # draw k12 edges for each node in S1
#         M = np.zeros((n1, n2))
#         targets = np.arange(n2)
#         for u in range(n1):
#             np.random.shuffle(targets)
#             vs = targets[:k12]
#             M[u, vs] = 1
#         return M
#     X = _sample_block(nA, nB, kAB)
#     Y = _sample_block(nA, nC, kAC)
#     Z = _sample_block(nB, nC, kBC)
#     A = np.bmat([[np.zeros((nA,nA)), X, Y],
#                  [np.transpose(X), np.zeros((nB,nB)), Z],
#                  [np.transpose(Y), np.transpose(Z), np.zeros((nC,nC))]])
#     g = nx.from_numpy_matrix(A)
#     b = dict(zip(range(0,nA), [0]*nA))
#     b.update(dict(zip(range(nA,nA+nB), [1]*nB)))
#     b.update(dict(zip(range(nA+nB,nA+nB+nC), [2]*nC)))
#     nx.set_node_attributes(g, 'tripartite', b)
#     return g


def tripartite_regular_configuration_graph(n, pA, pB, kAB, kAC, kBC,
                                           verbose=False):
    '''
    Generate a tripartite regular random graph using a stub-joining
    process.

    Parameters
    ==========
    n, int
      Number of nodes.
    pA, float, 0 <= pA <= 1
      Size of set A.
    pB, float, 0 <= pB <= 1
      Size of set B.
    kAB, float, kAB >= 0
      Degree from set A to B.
    kAC, float, kAC >= 0
      Degree from set A to C.
    kBC, float, kBC >= 0
      Degree from set B to C.

    The graph is generated by creating bipartite configuration model graphs
    for each pair of node sets (A,B), (A,C), (B,C).

    example
    =======

    import tripartite as tri
    import spectrum as spec
    import matplotlib.pyplot as plt
    g = tri.tripartite_regular_configuration_graph(2400, .2, .3, 3,5,5)
    L = spec.compute_spectrum(g)
    plt.hist(L, bins=301, normed=True)
    '''
    nA = int(np.round(n * pA))
    nB = int(np.round(n * pB))
    nC = n - nA - nB
    pC = 1-pA-pB
    kBA = int(np.round((pA/pB)*kAB))
    kCB = int(np.round((pB/pC)*kBC))
    kCA = int(np.round((pA/pC)*kAC))
    if verbose:
        print 'generating tripartite graph'
        print 'n = ', n
        print 'pA = ', pA
        print 'pB = ', pB
        print 'pC = ', pC
        print 'nA = ', nA
        print 'nB = ', nB
        print 'nC = ', nC
        print 'kAB = ', kAB
        print 'kAC = ', kAC
        print 'kBA = ', kBA
        print 'kBC = ', kBC
        print 'kCB = ', kCB
        print 'kCA = ', kCA
    def _extract_blocks(A, n1, n2):
        X = A[0:n1, n1:n1+n2]
        Xt = A[n1:n1+n2, 0:n1]
        return X, Xt
    g1 = nx.bipartite_configuration_model([kAB]*nA, [kBA]*nB)
    # while g1.is_multigraph():
    #     g1 = nx.bipartite_configuration_model([kAB]*nA, [kBA]*nB)
    X,Xt = _extract_blocks(nx.to_numpy_matrix(g1), nA, nB)
    g2 = nx.bipartite_configuration_model([kAC]*nA, [kCA]*nC)
    # while g2.is_multigraph():
    #     g2 = nx.bipartite_configuration_model([kAC]*nA, [kCA]*nC)
    Y,Yt = _extract_blocks(nx.to_numpy_matrix(g2), nA, nC)
    g3 = nx.bipartite_configuration_model([kBC]*nB, [kCB]*nC)
    # while g3.is_multigraph():
    #     g3 = nx.bipartite_configuration_model([kBC]*nB, [kCB]*nC)
    Z,Zt = _extract_blocks(nx.to_numpy_matrix(g3), nB, nC)
    A = np.bmat([[np.zeros((nA,nA)), X, Y],
                 [Xt, np.zeros((nB,nB)), Z],
                 [Yt, Zt, np.zeros((nC,nC))]])
    g = nx.from_numpy_matrix(A)
    b = dict(zip(range(0,nA), [0]*nA))
    b.update(dict(zip(range(nA,nA+nB), [1]*nB)))
    b.update(dict(zip(range(nA+nB,nA+nB+nC), [2]*nC)))
    nx.set_node_attributes(g, 'tripartite', b)
    return g
                      


