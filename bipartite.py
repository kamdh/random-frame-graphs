from framegraph.spectrum import compute_spectrum
from framegraph.frame import Frame
from framegraph.nonbacktracking import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

epsilon = 0.00001
offset = 0.5
num_vertex = 200
num_rep = 20

ps = {0: 0.7, 1: 0.3}
deg = {(0,1): 3, (1,0): 7}
xs = np.linspace(-10,10,701)
g = Frame(nx.DiGraph(nx.path_graph(2)), p=ps, deg=deg)
Aspec = g.spectrum(xs, epsilon=epsilon, offset=offset)
plt.ion()
plt.figure(0)
plt.plot(xs, np.abs(Aspec), 'b-', linewidth=2)
#raw_input()

# test guess in bulk
eig_arr = np.zeros(num_rep)
for r in range(num_rep):
    gsamp = g.sample(num_vertex, parallel_edges=True)
    Aspec_samp = compute_spectrum(gsamp)
    l_2 = np.sort(abs(Aspec_samp))[-3]
    eig_arr[r] = l_2
    print("solved rep %d, eig = %f" % (r,l_2))

plt.hist(Aspec_samp, bins = 121, normed = True)
plt.ylim((0,1))
plt.savefig("../figures/bipartite/spectrum_n_%d_k1_%d_k2_%d.png"
            %(num_vertex,deg[(0,1)],deg[(1,0)]))

plt.figure(1)
plt.hist(eig_arr)
#l_2_guess = 2*np.power((deg[(0,1)]-1)*(deg[(1,0)]-1), 0.25)
l_2_guess = np.sqrt(2*np.sqrt((deg[(0,1)]-1)*(deg[(1,0)]-1))
                  + deg[(0,1)] + deg[(1,0)]-2)
plt.vlines(l_2_guess,0,plt.ylim()[1],colors='r',lw=2)
print("|l_2|: emp = %f, guess = %f" %(l_2,l_2_guess))
plt.savefig("../figures/bipartite/l_2_n_%d_k1_%d_k2_%d.png"
            %(num_vertex,deg[(0,1)],deg[(1,0)]))

B_eigs = compute_spectrum(g.sample(num_vertex, parallel_edges=True),
                          matrix="nonbacktracking")
plt.figure()
plt.scatter(B_eigs.real, B_eigs.imag, alpha = 0.2)
