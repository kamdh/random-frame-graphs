import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh,eigvals

def choicemap(ks, ps):
    numk = len(ks)
    assert sum(ps) == 1.0 and len(ps) == numk
    assert len( np.unique( ks )) == numk
    return { ks[i]: ps[i] for i in range(numk) }

def gen_delta_graph(n, ks, ps, simple=False, model='conf'):
    '''
    Generate a graph whose degree distribution is of finite support.
    
    Parameters
    ----------
    n : number of nodes
    ks : list
        Degrees the nodes can have
    ps : list
        Probability for each edge, must have same length as ks and sum to 1
    '''
    # the following function generates a degree sequence of length n
    pkmap = choicemap(ks,ps)
    dseqfun = lambda n: [ nx.utils.weighted_choice(pkmap) for i in range(n) ]
    success = False
    while not success:
        dseq = dseqfun(n)
        if nx.is_valid_degree_sequence(dseq): 
            success = True
    if model == 'conf':
        G = nx.configuration_model(dseq)
    elif model == 'chung-lu':
        G = nx.expected_degree_graph(dseq)
    else:
        raise Exception('wrong graph model requested')
    if simple:
        G = nx.Graph(G) # removes multi-edges
        G.remove_edges_from( G.selfloop_edges() ) # remove self loop
    return G

def adjacency_spectrum_simple(G, weight):
    '''
    Compute the spectrum for the adjacency matrix.
    Faster than the built-in routine for networkx because it uses a
    symmetric solver and disables some of the checks.
    '''
    assert isinstance(G, nx.classes.Graph)
    return eigvalsh(nx.adjacency_matrix(G, weight=weight).todense(), 
                    overwrite_a=True, check_finite=False)

def compute_spectrum(G, matrix=None, weight='weight'):
    if matrix is None:
        #spec = nx.linalg.spectrum.adjacency_spectrum(G, weight)
        spec = adjacency_spectrum_simple(G, weight)
    elif matrix == 'laplacian':
        spec = nx.linalg.spectrum.laplacian_spectrum(G, weight)
    elif matrix == 'nonbacktracking':
        B = non_backtracking_matrix(G)
        spec = eigvals(B.todense(), overwrite_a=True, check_finite=False)
    return spec

def normal_complex(a, b, n):
    return np.random.normal(a,b,n)+1.0j*np.random.normal(a,b,n)

def resolvent_y_iter_mc(z, ks, ps, alpha, 
                        s0=normal_complex(0,10,20), maxiter=100):
    ## setup Q(k)
    qs = ks * ps / np.sum(ks * ps)
    qks = ks - 1
    numk = len(ks)
    ## initialize distribution
    s = s0
    numparticles = s.shape[0]
    for i in range(maxiter):
        # N = np.array( [ nx.utils.weighted_choice( choicemap(qks, qs) ) \
        #                 for i in range(numparticles) ] )
        N = np.array( np.random.choice(qks, numparticles, p=qs) )
        sumSamps = np.array( map( lambda n: np.sum(np.random.choice(s, n)), N ) )
        s = -( z + alpha * (N + 1) + sumSamps )**-1
    return s

def resolvent_x_iter_mc(z, y, ks, ps, alpha,
                        s0=normal_complex(0,10,20), maxiter=100):
    ## initialize distribution
    s = s0
    numparticles = s.shape[0]
    for i in range(maxiter):
        # N = np.array( [ nx.utils.weighted_choice( choicemap(ks, ps) ) \
        #                 for i in range(numparticles) ] )
        N = np.array( np.random.choice(ks, numparticles, p=ps) )
        sumSamps = np.array( map( lambda n: np.sum(np.random.choice(y, n)), N ) )
        s = -( z + alpha * N + sumSamps )**-1
    return s

def resolvent_xavg_mc(z, ks, ps, alpha, maxiter=100):
    y = resolvent_y_iter_mc(z, ks, ps, alpha, 
                            s0=np.random.normal(0,10,400), maxiter=maxiter)
    x = resolvent_x_iter_mc(z, y, ks, ps, alpha, maxiter=maxiter)
    xavg = np.mean(x)
    return xavg

def spectrum_mc(xs, epsilon, ks, ps, alpha):
    z = np.array( map(lambda x: 
                      resolvent_xavg_mc(x + 1.0j*epsilon, 
                                        ks, ps, alpha, maxiter=100), xs) )
    mu = np.imag( z ) / np.pi
    return mu

def resolvent_y_iter_analytic(z, ks, ps, alpha, s0, maxiter):
    # tol = 1e-8
    qs = ks * ps / np.sum(ks * ps)
    qks = ks - 1
    numk = len(ks)
    snext = lambda s, z: -( z + alpha * (qks + 1) + \
                                  qks * np.sum( qs * s ) )**-1
    s = s0
    for i in range(maxiter):
        snew = snext(s, z)
        # if np.max( np.abs( snew - s ) ) < tol:
        #     snew = s
        #     break
        # else:
        #     snew = s
    return s

def resolvent_y(z, ks, ps, alpha, maxiter=100):
    assert np.imag(z) != 0, ("resolvent should be called with "
                             "nonzero imaginary part")
    s0 = np.array( [z for i in range(len(ks))] )
    return resolvent_y_iter_analytic(z, ks, bps, alpha, s0, maxiter)

def resolvent_x(z, yz, ks, ps, alpha):
    assert np.imag(z) != 0, ("resolvent should be called with "
                             "nonzero imaginary part")
    qs = ks * ps / np.sum(ks * ps)
    qks = ks + 1
    numk = len(ks)
    return -( z + alpha * ks + ks * np.sum( qs * yz ) )**-1

def resolvent_xtot(z, ks, ps, alpha, maxiter=100):
    assert np.imag(z) != 0, ("resolvent should be called with "
                             "nonzero imaginary part")
    yz = resolvent_y(z, ks, ps, alpha, maxiter)
    xz = resolvent_x(z, yz, ks, ps, alpha)
    xtot = np.sum( ps*xz )
    return xtot

def spectrum_analytic(xs, epsilon, ks, ps, alpha):
    z = np.array( map(lambda x: 
                      resolvent_xtot(x + 1.0j*epsilon, 
                                     ks, ps, alpha, maxiter=100), xs) )
    mu = np.imag( z ) / np.pi
    return mu


def non_backtracking_matrix(G, oriented=True):
    num_E=G.number_of_edges()
    if oriented:
        B=np.zeros((2*num_E,2*num_E))
        edge_list=G.edges()
        # append other orientation
        edge_list = edge_list + [(e[1],e[0]) for e in edge_list]
        for i,e in enumerate(edge_list):
            for j,f in enumerate(edge_list):
                if e[1]==f[0] and e[0]!=f[1]:
                    B[i,j]=1
    else:
        B=np.zeros((num_E,num_E))
        edge_list=G.edges()
        for i,e in enumerate(edge_list):
            for j,f in enumerate(edge_list):
                if e[1]==f[0] and e[0]!=f[1]:
                    B[i,j]=1
    return B
