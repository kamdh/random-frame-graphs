import networkx as nx
from networkx.exception import NetworkXError
from networkx.algorithms.bipartite import random_graph \
     as bipartite_random_graph
from warnings import warn
# from IPython import embed
import numpy as np 

class SBM(nx.DiGraph):
    '''
    SBM class. The additional arguments of the class are:
    
    Parameters
    ----------
    p : list of proportions for the node sets of the frame, should sum to 1
    deg : expected degrees of the edge set, dictionary with keys (node1, node2)
         and values degree12
    '''
    def __init__(self, data=None, p={}, deg={}):
        super(SBM,self).__init__(data)
        nx.set_node_attributes(self, 'p', p)
        nx.set_edge_attributes(self, 'deg', deg)
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
        for e in self.edges_iter():
            u = e[0]
            v = e[1]
            k_uv = self.adj[u][v]['deg']
            if self.has_edge(v,u):
                # check detailed balance
                k_vu = self.adj[v][u]['deg']
                pu = self.node[u]['p']
                pv = self.node[v]['p']
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
    
    def _yfun_iterate(self, y, z, n_replica, alpha):
        '''
        Iteration of the Y equations right hand side
        '''
        assert y.shape == (self.number_of_edges(),n_replica)
        yout = np.zeros(y.shape, dtype=np.complex)
        edgelist = self.edges()
        # # pick column to update
        # col_idx=np.random.randint(n_replica)
        for col_idx in range(n_replica):
            # # remaining columns
            # other_col=np.setdiff1d(np.arange(n_replica),[col_idx])
            other_col=range(n_replica)
            for idx, e in enumerate(edgelist):
                (u,v) = e
                sumval = 0.0
                for w in self.neighbors(v):
                    widx = edgelist.index((v,w))
                    k_vw = self.adj[v][w]['deg']
                    if w == u:
                        # select excess degree
                        k_samp=_sample_excess_degree(k_vw)
                    else:
                        # select degree
                        k_samp=_sample_degree(k_vw)
                    # select neighbors
                    neighbor_idx=np.random.choice(other_col,
                                                  k_samp,
                                                  replace=True)
                    for col in neighbor_idx:
                        sumval += y[widx,col]
                    sumval += k_samp*alpha
                yout[idx,col_idx] = -1.0/(z + sumval)
        return yout

    def _xfun_iterate(self, y, z, n_replica, alpha):
        '''
        This function is the resolvent evaluated at a node (X equations).
        '''
        assert y.shape == (self.number_of_edges(),n_replica)
        xout = np.zeros((self.number_of_nodes(),n_replica), dtype=np.complex)
        nodelist = self.nodes()
        edgelist = self.edges()
        # # pick column to update
        # col_idx=np.random.randint(n_replica)
        # # remaining columns
        # other_col=np.setdiff1d(np.arange(n_replica),[col_idx])
        for col_idx in range(n_replica):
            # # remaining columns
            # other_col=np.setdiff1d(np.arange(n_replica),[col_idx])
            other_col=range(n_replica)        
            for idx, v in enumerate(nodelist):
                sumval = 0.0
                for w in self.neighbors(v):
                    widx = edgelist.index((v,w))
                    k_vw = self.adj[v][w]['deg']
                    # sample degree
                    k_samp = _sample_degree(k_vw)
                    # select neighbors
                    neighbor_idx=np.random.choice(other_col,k_samp,
                                                  replace=True)
                    for col in neighbor_idx:
                        sumval += y[widx,col]
                    sumval += alpha*k_samp
                xout[idx,col_idx] = -1.0/(z + sumval)
        return xout

    def spectrum(self, xs, n_replica=100,
                 epsilon=0.01, alpha=0, offset=1.0,
                 max_iterates=100, transient=50):
        assert self.detailed_balance(), 'detailed balance not satisfied'
        from scipy.optimize import root
        N = self.number_of_edges()
        density_vec = np.zeros(xs.shape)
        y0 = offset*np.ones((N,1)) + offset*np.ones((N,1))*1.0j
        y_pop = np.tile(y0,(1,n_replica))
        x_pop = np.zeros((self.number_of_nodes(),n_replica),dtype=np.complex)
        y_avg = np.zeros((N,n_replica),dtype=np.complex)
        x_avg = np.zeros((self.number_of_nodes(),n_replica),dtype=np.complex)
        n=0
        for idx, x in enumerate(xs):
            z = x + epsilon*1.0j
            for iterate in range(max_iterates):
                y_pop = self._yfun_iterate(y_pop, z, n_replica, alpha)
                #x_pop = self._xfun_iterate(y_pop, z, n_replica, alpha)
                if (iterate > transient):
                    n += 1
                    y_avg = y_avg + (y_pop - y_avg)/n
            #x_pop = self._xfun_iterate(y_pop, z, n_replica, alpha)
            x_pop = self._xfun_iterate(y_avg, z, n_replica, alpha)
            #import pdb; pdb.set_trace()
            #x_soln = np.mean(x_avg,axis=1)
            x_soln = np.mean(x_pop,axis=1)
            # print "y"
            # print y_pop
            # print "avg x"
            # print x_soln
            ps = [d['p'] for n,d in self.nodes_iter(data=True)]
            density = np.imag(np.dot(ps, x_soln))/np.pi
            print "density(%0.3f) = %f" % (x,density)
            density_vec[idx] = density
        return density_vec

    def base_matrices(self):
        P=np.matrix(np.diag([d['p'] for n,d in self.nodes_iter(data=True)]))
        K=nx.linalg.adjacency_matrix(self, weight='deg').todense()
        Q=P*K/np.tile(np.sum(P*K,axis=1),(1,self.number_of_nodes()))
        return P,K,Q

    def sample(self,n):
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
        for u in self.nodes_iter():
            p = self.node[u]['p']
            try:
                np.testing.assert_almost_equal(p*n, int(p*n))
            except AssertionError:
                print("n*p is not integer for node %d, n=%f" % (u,n))
                raise NetworkXError('frame is not realizable')
            n_block[u] = int(p*n)
        n_blocksum=np.cumsum(n_block)
        # fill in adj_mat
        traversed={}
        for e in self.edges_iter():
            traversed[e]=1
            u = e[0]
            v = e[1]
            k_uv = self.adj[u][v]['deg']
            k_vu = self.adj[v][u]['deg']
            n_u = int(n*self.node[u]['p'])
            n_v = int(n*self.node[v]['p'])
            if u == v:
                # on-diagonal block
                g1 = nx.generators.fast_gnp_random_graph(n_u, float(k_uv)/n_u)
                X = np.array(nx.to_numpy_matrix(g1))
                if u == 0:
                    i_lower=0
                else:
                    i_lower=n_blocksum[u-1]
                adj_mat[i_lower:n_blocksum[u],i_lower:n_blocksum[u]] = X
            elif not traversed.has_key((v,u)):
                # off-diagonal block
                g1 = bipartite_random_graph(n_u, n_v, float(k_uv)/n_v)
                X,Xt = _extract_blocks(nx.to_numpy_matrix(g1), n_u, n_v)
                if u == 0:
                    i_lower=0
                else:
                    i_lower=n_blocksum[u-1]                
                if v == 0:
                    j_lower=0
                else:
                    j_lower=n_blocksum[v-1]                
                adj_mat[i_lower:n_blocksum[u],j_lower:n_blocksum[v]] = X
                adj_mat[j_lower:n_blocksum[v],i_lower:n_blocksum[u]] = Xt
        adj_mat=np.matrix(adj_mat)
        # adj_mat now filled in, so generate graph
        g = nx.from_numpy_matrix(adj_mat)
        b = dict(zip(range(0,n_blocksum[0]), [0]*n_blocksum[0]))
        for i in range(1,self.number_of_nodes()):
            b.update(dict(zip(range(n_blocksum[i-1],n_blocksum[i]),
                              [i]*n_block[i])))
        nx.set_node_attributes(g, 'block', b)
        return g
        
def _sample_degree(k):
    return np.random.poisson(lam=k)

def _sample_excess_degree(k):
    return np.random.poisson(lam=k)
