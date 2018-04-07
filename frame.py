import networkx as nx
from networkx.exception import NetworkXError
from networkx.algorithms.bipartite import configuration_model \
     as bipartite_configuration_model
from warnings import warn
# from IPython import embed
import numpy as np 

class Frame(nx.DiGraph):
    '''
    This class defines a k-frame random graph, which is a random graph with
    an underlying digraph structure, the "frame". Each node in the frame
    is a set of nodes in the resulting random graph, and the edges in the
    frame represent all the possible edges in the resulting graph. The 
    additional arguments of the class are:
    
    Parameters
    ----------
    p : list of proportions for the node sets of the frame, should sum to 1
    deg : degrees of the node sets, dictionary with keys (node1, node2)
         and values degree12
    '''
    def __init__(self, data=None, p={}, deg={}):
        super(Frame,self).__init__(data)
        if set(self.nodes()) != set(p.keys()):
            raise NetworkXError('node set does not match keys in p dict')
        nx.set_node_attributes(self, p, name='p')
        if set(self.edges()) != set(deg.keys()):
            raise NetworkXError('edge set does not match keys in deg dict')
        nx.set_edge_attributes(self, deg, name='deg')
        if not self.detailed_balance():
            raise NetworkXError('detailed balance not satisfied')
        
    def detailed_balance(self):
        '''
        Check the detailed balance conditions for the given Frame.
        These are that p[u] deg[u,v] == p[v] deg[v,u],
        where p and deg are the node proportion and degree attributes, 
        respectively.

        Returns
        -------
          True or False
        '''
        for e in self.edges():
            u = e[0]
            v = e[1]
            k_uv = self.adj[u][v]['deg']
            if self.has_edge(v,u):
                # check detailed balance
                k_vu = self.adj[v][u]['deg']
                pu = self.nodes[u]['p']
                pv = self.nodes[v]['p']
                try:
                    np.testing.assert_approx_equal(pu*k_uv,pv*k_vu)
                except AssertionError:
                    print ('detailed balance not satisfied for edges %s and %s '
                           'and possibly others') % (str((u,v)), str((v,u)))
                    return False
            else:
                # it cannot be satisfied if edges aren't reciprocated
                print ('detailed balance cannot be satisfied for '
                       'unreciprocated edges, %s missing') % str((v,u))
                return False
        return True
    
    def _yfun_root(self, y, z, alpha):
        '''
        The root of this function is the solution of the system of
        quadratic distributional equations (Y equations).
        '''
        assert y.shape == (self.number_of_edges(),)
        yout = np.zeros(self.number_of_edges(), dtype=np.complex)
        edgelist = list(self.edges())
        for idx, e in enumerate(edgelist):
            (u,v) = e
            sumval = 0.0
            for w in self.neighbors(v):
                widx = edgelist.index((v,w))
                yvw = y[widx]
                kvw = self.adj[v][w]['deg']
                if w == u:
                    delta = 1
                else:
                    delta = 0
                sumval += yvw*(kvw - delta) + kvw*alpha
            yout[idx] = 1 + y[idx]*(z + sumval)
        return yout

    def _yfun_root_real(self, y, z, alpha):
        '''
        Wrapper function which takes real and imaginary parts 
        (0:N and N:2*N, respectively) and passes them to the complex function,
        returning the real and imaginary parts. This allows the function to
        be passed to the Fortran root-finding algorithms for real functions
        on the reals.
        '''
        N = self.number_of_edges()
        assert y.shape == (2*N,)
        y_complex = y[0:N] + 1.0j*y[N:2*N]
        y_eval = self._yfun_root(y_complex, z, alpha)
        return np.hstack((y_eval.real, y_eval.imag))

    def _xfun(self, y, z, alpha):
        '''
        This function is the resolvent evaluated at a node (X equations).
        '''
        assert y.shape == (self.number_of_edges(),)
        xout = np.zeros(self.number_of_nodes(), dtype=np.complex)
        nodelist = self.nodes()
        edgelist = list(self.edges())
        for idx, v in enumerate(nodelist):
            sumval = 0.0
            for w in self.neighbors(v):
                widx = edgelist.index((v,w))
                yvw = y[widx]
                kvw = self.adj[v][w]['deg']
                sumval += (yvw + alpha)*kvw
            xout[idx] = -1./(z + sumval)
        return xout

    def spectrum(self, xs, epsilon=0.01, alpha=0, offset=1.0):
        assert self.detailed_balance(), 'detailed balance not satisfied'
        from scipy.optimize import root
        N = self.number_of_edges()
        density_vec = np.zeros(xs.shape)
        y0 = np.hstack((np.zeros(N),
                        offset*np.ones(N)))
        for idx, x in enumerate(xs):
            z = x + epsilon*1.0j
            y_soln_real = root(self._yfun_root_real, y0, args=(z, alpha)).x
            y_soln_cmplx = y_soln_real[0:N] + 1.0j*y_soln_real[N:2*N]
            x_soln = self._xfun(y_soln_cmplx, z, alpha)
            ps = [d['p'] for n,d in self.nodes(data=True)]
            density = np.imag(np.dot(ps, x_soln))/np.pi
            density_vec[idx] = density
        return density_vec

    def base_matrices(self):
        Pdiag = np.matrix(np.diag([d['p'] for n,d in self.nodes(data=True)]))
        K = nx.linalg.adjacency_matrix(self, weight='deg').todense()
        Q = Pdiag * K / \
          np.tile(np.sum(Pdiag*K,axis=1),
                  (1,self.number_of_nodes())).astype(float)
        P = K/np.tile(np.sum(K, axis=1), (1,self.number_of_nodes())).astype(float)
        return Pdiag, K, Q, P

    def sample(self, n, parallel_edges=True):
        '''
        Sample a random graph from the frame family.

        Parameters
        ----------
          n : number of nodes in resulting graph
        
        Returns
        -------
          nx.Graph
        '''

        def _extract_blocks(A, n1, n2):
            X = A[0:n1, n1:n1+n2]
            Xt = A[n1:n1+n2, 0:n1]
            return X, Xt

        adj_mat = np.zeros((n,n))
        n_block = np.zeros(self.number_of_nodes(),dtype=int)
        # check realizability and fill n_block
        for u in self.nodes():
            p = self.nodes[u]['p']
            try:
                np.testing.assert_almost_equal(p*n, int(p*n))
            except AssertionError:
                print("n*p is not integer for node %d, n=%f" % (u,n))
                raise NetworkXError('frame is not realizable')
            n_block[u] = int(p*n)
        n_blocksum=np.cumsum(n_block)
        # fill in adj_mat
        traversed={}
        for e in self.edges():
            traversed[e]=1
            u = e[0]
            v = e[1]
            k_uv = self.adj[u][v]['deg']
            k_vu = self.adj[v][u]['deg']
            n_u = int(n*self.nodes[u]['p'])
            n_v = int(n*self.nodes[v]['p'])
            if u == v:
                # on-diagonal block
                reg_graph = nx.generators.random_regular_graph(k_uv,n_u)
                X = np.array(nx.to_numpy_matrix(reg_graph))
                if u == 0:
                    i_lower=0
                else:
                    i_lower=n_blocksum[u-1]
                adj_mat[i_lower:n_blocksum[u],i_lower:n_blocksum[u]] = X
            elif not (v,u) in traversed:
                # off-diagonal block
                g1 = bipartite_configuration_model([k_uv]*n_u, [k_vu]*n_v)
                X,Xt = _extract_blocks(nx.to_numpy_matrix(g1), n_u, n_v)
                if u == 0:
                    i_lower=0
                else:
                    i_lower=n_blocksum[u-1]
                if v == 0:
                    j_lower=0
                else:
                    j_lower=n_blocksum[v-1]
                adj_mat[i_lower:n_blocksum[u], j_lower:n_blocksum[v]] = X
                adj_mat[j_lower:n_blocksum[v], i_lower:n_blocksum[u]] = Xt
        adj_mat = np.matrix(adj_mat)
        # adj_mat now filled in, so generate graph
        g = nx.from_numpy_matrix(adj_mat, parallel_edges=parallel_edges)
        # set block attributes
        b = dict(zip(range(0,n_blocksum[0]), [0]*n_blocksum[0]))
        for i in range(1,self.number_of_nodes()):
            b.update(dict(zip(range(n_blocksum[i-1],n_blocksum[i]),
                              [i]*n_block[i])))
        nx.set_node_attributes(g, b, name='block')
        return g
        
