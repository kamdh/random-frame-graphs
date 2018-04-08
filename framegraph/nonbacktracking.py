import networkx as nx
from networkx.utils import not_implemented_for

@not_implemented_for('multigraph')
def nonbacktracking_matrix(G, weight=None):
    """Return non-backtracking matrix of G.

    The non-backtracking matrix [1]_ is indexed by the oriented edges of G,
    and it is defined by

    .. math::

        B_{e,f} = \mathcal{1}(e_2 = f_1) \mathcal{1}(e1 \neq f_2)

    where `e=(e_1,e_2)` and `f=(f_1,f_2)` are directed edges of G.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1. Weights are summed along edges
       in the path, i.e. `ef` in the example. 

    Returns
    -------
    B : SciPy sparse matrix
       The non-backtracking matrix of G.

    References
    ----------
    .. [1] K-I. Hashimoto, Zeta functions of finite graphs and representations
       of p-adic groups, Adv. Stud. Pure Math., Volume 15, pp. 211-280, 1989.
    """
    import scipy.sparse
    if G.is_directed():
        Gd = G
    else:
        Gd = nx.DiGraph(G) # forms the directed edges
    edgelist = list(Gd.edges())
    B = scipy.sparse.lil_matrix((len(edgelist),len(edgelist)))
    edge_index = dict( (edge[:2],i) for i,edge in enumerate(edgelist) )
    for ei,e in enumerate(edgelist):
        (e1,e2) = e[:2]
        for f2 in Gd.successors(e2):
            if f2 != e1:
                # then it doesn't backtrack
                fi = edge_index[(e2,f2)]
                if weight is None:
                    wt = 1
                else:
                    wt = G[e1][e2].get(weight,1) + G[e2][f2].get(weight,1)
                B[ei,fi] = wt
    return B.asformat('csc')

def nonbacktracking_matrix_guess(G):
    """Return non-backtracking matrix of G.
    """
    import scipy.sparse
    import networkx as nx
    import numpy as np
    if G.is_directed():
        Gd = G
    else:
        Gd = nx.DiGraph(G) # forms the directed edges
    edgelist = list(Gd.edges())
    B = scipy.sparse.lil_matrix((len(edgelist),len(edgelist)))
    edge_index = dict( (edge[:2],i) for i,edge in enumerate(edgelist) )
    P = np.matrix(np.diag([d['p'] for n,d in Gd.nodes(data = True)]))
    K = nx.linalg.adjacency_matrix(Gd, weight = 'deg').todense()
    Q = P*(K+K.T)/np.tile(np.sum(P*(K+K.T),axis = 1),(1,K.shape[1]))
    for ei,e in enumerate(edgelist):
        (e1,e2) = e[:2]
        for f2 in Gd.successors(e2):
            fi = edge_index[(e2,f2)]
            wt = K[e2,f2]
            B[ei,fi] = wt
    return B.asformat('csc')
