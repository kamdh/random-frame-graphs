import networkx as nx
from networkx.exception import NetworkXError
from warnings import warn
from IPython import embed
import numpy as np 

class Frame(nx.DiGraph):
    def __init__(self, data=None, p={}, deg={}):
        super(Frame,self).__init__(data)
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
        =======
          True or False
        '''
        tol = 1e-10
        for e in self.edges_iter():
            u = e[0]
            v = e[1]
            kuv = self.adj[u][v]['deg']
            if self.has_edge(v,u):
                # check detailed balance
                kvu = self.adj[v][u]['deg']
                pu = self.node[u]['p']
                pv = self.node[v]['p']
                if abs(pu*kuv - pv*kvu) > tol:
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
        quadratic distributional equations Y.
        '''
        assert y.shape == (self.number_of_edges(),)
        yout = np.zeros(self.number_of_edges(), dtype=np.complex)
        edgelist = self.edges()
        for idx, e in enumerate(edgelist):
            (u,v) = e
            sumval = 0
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
        returning the real and imaginary parts. This allows the function
        to be passed to the Fortran root-finding algorithms.
        '''
        N = self.number_of_edges()
        assert y.shape == (2*N,)
        y_complex = y[0:N] + 1.0j*y[N:2*N]
        y_eval = self._yfun_root(y_complex, z, alpha)
        return np.hstack((y_eval.real, y_eval.imag))

    def _xfun(self, y, z, alpha):
        assert y.shape == (self.number_of_edges(),)
        xout = np.zeros(self.number_of_nodes(), dtype=np.complex)
        nodelist = self.nodes()
        edgelist = self.edges()
        for idx, v in enumerate(nodelist):
            sumval = 0
            for w in self.neighbors(v):
                widx = edgelist.index((v,w))
                yvw = y[widx]
                kvw = self.adj[v][w]['deg']
                sumval += (yvw + alpha)*kvw
            xout[idx] = -1./(z + sumval)
        return xout

    def spectrum(self, xs, epsilon=0.005, alpha=0, offset=1):
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
            ps = [d['p'] for n,d in self.nodes_iter(data=True)]
            density = np.imag(np.dot(ps, x_soln))/np.pi
            density_vec[idx] = density
        return density_vec
