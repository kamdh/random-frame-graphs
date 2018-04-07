#!/usr/bin/env python
from spectrum import compute_spectrum
from sbm import SBM
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

epsilon=1e-4
offset=1
num_vertex=4000
num_rep=20

## 2-path
ps = {0: 0.2, 1: 0.8}
deg = {(0,1): 18, (1,0): 4.5}
g = SBM(nx.DiGraph(nx.path_graph(2)), p=ps, deg=deg)
xs = np.linspace(-8,8,91)

# ## 2-group SBM
# ps = {0: 0.8, 1: 0.2}
# deg = {(0,1): 9, (1,0): 36, (0,0): 3, (1,1):4}
# g1 = nx.DiGraph(nx.complete_graph(2))
# g1.add_edge(0,0)
# g1.add_edge(1,1)
# g = SBM(g1, p=ps, deg=deg)
# xs = np.linspace(-11,11,201)

# ## 3-group SBM
# pA = 0.125
# pB = 0.125
# kAB = 3
# kAC = 6
# kBC = 12
# ps = {0: pA, 1: pB, 2: 1-pA-pB}
# deg = {(0,1): kAB, (0,2): kAC, (1,2): kBC,
#        (1,0): kAB*pA/pB, (2,0): kAC*pA/(1-pA-pB), (2,1): kBC*pB/(1-pA-pB)}
# g1 = nx.DiGraph(nx.complete_graph(3))
# g = SBM(g1, p=ps, deg=deg)
# xs = np.linspace(-6.1,6.1,121)

Aspec = g.spectrum(xs, n_replica=120, offset=offset,
                   epsilon=epsilon, y_max_iter=80,y_transient=50,
                   x_max_iter=40)

gsamp=g.sample(num_vertex)
Aspec_samp = compute_spectrum(gsamp)

plt.ion()
plt.figure(0)
plt.hist(Aspec_samp, bins=91, normed=True)
plt.hold(True)
plt.plot(xs, np.abs(Aspec), 'k-', linewidth=2)
plt.ylim((0,0.15))
plt.savefig('test.png')

plt.figure(1)
plt.spy(nx.adjacency_matrix(gsamp), marker='.')

#raw_input()
