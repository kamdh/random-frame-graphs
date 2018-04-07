import networkx as nx
from networkx.exception import NetworkXError
from networkx.algorithms.bipartite import random_graph \
     as bipartite_random_graph
from warnings import warn
# from IPython import embed
import numpy as np
#from pathos.multiprocessing import ProcessingPool
import pymp

class SBM(nx.DiGraph):
    '''
    SBM class. The additional arguments of the class are:
    
    Parameters
    ----------
    p : list of proportions for the node sets of the frame, should sum to 1
    deg : expected degrees of the edge set, dictionary with keys (node1, node2)
         and values degree12
    '''

    #num_threads = 4
    
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
        MCMC iteration of the Y equations
        '''
        sample_neighbors = lambda k, n: \
          (np.random.uniform(size=k) * n).astype(int)

        def _update_col(self, y, z, n_replica, alpha, edgelist):
            y_col = np.zeros((y.shape[0],1), dtype=np.complex)
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
                    # neighbor_idx=np.random.choice(other_col,
                    #                               k_samp,
                    #                               replace=True)
                    # neighbor_idx = np.random.multinomial(n_replica,
                    #                                      p=probs,
                    #                                      size=k_samp)
                    neighbor_idx = sample_neighbors(k_samp, n_replica)
                    for col in neighbor_idx:
                        sumval += y[widx,col]
                    sumval += k_samp*alpha
                y_col[idx] = -1.0/(z + sumval)
            return y_col
        
        assert y.shape == (self.number_of_edges(), n_replica)
        yout = [_update_col(self, y, z, n_replica, alpha,
                            self.edges())
                for col in range(n_replica)]
        yout = np.hstack(tuple(yout))
        # ## pymp parallel version
        # yout = pymp.shared.array((y.shape[0], n_replica), dtype=np.complex)
        # with pymp.Parallel() as p:
        #     for col in p.xrange(n_replica):
        #         yout[:,col] = _update_col(self, y, z, n_replica, alpha,
        #                                   self.edges()).flatten()
        # ## pathos parallel version
        # pool = ProcessingPool(self.num_threads)
        # yout = pool.map(lambda x:
        #                 _update_col(self, y, z, n_replica, alpha, edgelist),
        #                   range(n_replica))
        # yout = np.hstack(tuple(yout))
        #pool.close()
        return yout

    def _xfun_iterate(self, y, z, n_replica, n_iter, alpha):
        '''
        MCMC iteration of the X equations
        '''
        sample_neighbors = lambda k, n: \
          (np.random.uniform(size=k) * n).astype(int)

        def _update_col(self, y, z, n_replica, alpha, nodelist, edgelist):
            x_col = np.zeros((self.number_of_nodes(),1), dtype=np.complex)
            #other_col=range(n_replica)        
            for idx, v in enumerate(nodelist):
                sumval = 0.0
                for w in self.neighbors(v):
                    widx = edgelist.index((v,w))
                    k_vw = self.adj[v][w]['deg']
                    # sample degree
                    k_samp = _sample_degree(k_vw)
                    # select neighbors
                    # neighbor_idx=np.random.choice(other_col,k_samp,
                    #                               replace=True)
                    # neighbor_idx = np.random.multinomial(n_replica,
                    #                                      p=probs,
                    #                                      size=k_samp)
                    neighbor_idx = sample_neighbors(k_samp,n_replica)
                    for col in neighbor_idx:
                        sumval += y[widx,col]
                    sumval += alpha*k_samp
                x_col[idx] = -1.0/(z + sumval)
            return x_col
            
        assert y.shape == (self.number_of_edges(), n_replica, n_iter)
        y = np.reshape(y, (self.number_of_edges(), n_replica * n_iter))
        xout = [_update_col(self, y, z, n_replica * n_iter, alpha,
                            self.nodes(), self.edges())
                for col in range(n_replica)]
        xout = np.hstack(tuple(xout))
        # ## pymp parallel version
        # xout = pymp.shared.array((self.number_of_nodes(), n_replica),
        #                           dtype=np.complex)
        # with pymp.Parallel() as p:
        #     for col in p.xrange(n_replica):
        #         xout[:,col] = _update_col(self, y, z, n_replica, alpha,
        #                                   self.nodes(), self.edges()).flatten()
        ## pathos parallel version
        # pool = ProcessingPool(self.num_threads)
        # xout = pool.map(lambda x:
        #                 _update_col(self, y, z, n_replica, alpha,
        #                             nodelist, edgelist),
        #                   range(n_replica))
        # xout = np.hstack(tuple(xout))
        #pool.close()
        return xout

    def spectrum(self, xs, n_replica=100, epsilon=0.01, alpha=0, offset=1.0,
                 y_max_iter=100, y_transient=50,
                 x_max_iter=10, transient_delay=1,
                 parallel=True):
        assert self.detailed_balance(), 'detailed balance not satisfied'
        assert y_max_iter > y_transient, 'y_max_iter should be > y_transient'
        # setup some vars
        N = self.number_of_edges()
        y0 = offset*np.ones((N,1)) + offset*np.ones((N,1))*1.0j
        ps = [d['p'] for n,d in self.nodes_iter(data=True)]
        # parallel compute density at each point x
        density_vec = pymp.shared.array(xs.shape, dtype='float64')
        with pymp.Parallel(if_=parallel) as p:
            for idx in p.xrange(len(xs)):
                x = xs[idx]
                y_pop = np.tile(y0,(1,n_replica))
                x_pop = np.zeros((self.number_of_nodes(),n_replica),
                                 dtype=np.complex)
                y_avg = np.zeros((N, n_replica, int(y_max_iter - y_transient)),
                                 dtype=np.complex)
                x_avg = np.zeros((self.number_of_nodes(),
                                  n_replica,
                                  x_max_iter),
                                 dtype=np.complex)
                z = x + epsilon*1.0j
                for i in range(y_transient):
                    y_pop = self._yfun_iterate(y_pop, z, n_replica, alpha)
                for i in range(y_max_iter - y_transient):
                    y_pop = self._yfun_iterate(y_pop, z, n_replica, alpha)
                    y_avg[:,:,i] = y_pop
                for i in range(x_max_iter):
                    x_pop = self._xfun_iterate(y_avg, z, n_replica,
                                               y_avg.shape[2], alpha)
                    x_avg[:,:,i] = x_pop
                #x_pop = self._xfun_iterate(y_pop, z, n_replica, alpha)
                #x_pop = self._xfun_iterate(y_avg, z, n_replica, alpha)
                #import pdb; pdb.set_trace()
                x_soln = np.mean(np.reshape(x_avg,
                                            (self.number_of_nodes(),
                                             n_replica * x_max_iter)),
                                 axis=1)
                #x_soln = np.mean(x_pop,axis=1)
                # print "y"
                # print y_pop
                # print "avg x"
                # print x_soln
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
