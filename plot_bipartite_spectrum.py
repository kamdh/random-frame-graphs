from framegraph.spectrum import compute_spectrum
from framegraph.frame import Frame
from framegraph.nonbacktracking import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

epsilon = 0.00001
offset = 0.5
num_vertex = 2000
num_rep = 20

ps = {0: 0.7, 1: 0.3}
deg = {(0,1): 3, (1,0): 7}
g = Frame(nx.DiGraph(nx.path_graph(2)), p=ps, deg=deg)

# l_2_guess = np.sqrt( 2 * np.sqrt( (deg[(0,1)] - 1) * (deg[(1,0)] - 1) )
#                     + deg[(0,1)] + deg[(1,0)] - 2)
l_2_guess = np.sqrt(deg[(0,1)] - 1) + np.sqrt(deg[(1,0)] - 1)
l_1 = np.sqrt( deg[(0,1)] * deg[(1,0)] )
bulk_min = - np.sqrt(deg[(0,1)] - 1) + np.sqrt(deg[(1,0)] - 1)

# Compute bulk spectrum
xneg = np.linspace(-l_2_guess, -bulk_min, 100)
xpos = np.linspace(bulk_min, l_2_guess, 100)
spec_neg = g.spectrum(xneg, epsilon=epsilon, offset=offset)
spec_pos = g.spectrum(xpos, epsilon=epsilon, offset=offset)

plt.ion()
plt.figure(0)
plt.plot(xneg, np.abs(spec_neg), 'k-', linewidth=1)
plt.plot(xpos, np.abs(spec_pos), 'k-', linewidth=1)

# test guess in bulk
eig_arr = np.zeros(num_rep)
for r in range(num_rep):
    gsamp = g.sample(num_vertex, parallel_edges=True)
    Aspec_samp = compute_spectrum(gsamp)
    l_2 = np.sort(abs(Aspec_samp))[-3]
    eig_arr[r] = l_2
    print("solved rep %d, eig = %f" % (r,l_2))

plt.hist(Aspec_samp, bins = 121, normed = True)
plt.vlines(l_2_guess, 0, plt.ylim()[1],
           colors='r', linestyles='dashed', lw=1)
plt.vlines(l_1, 0, plt.ylim()[1],
           colors='r', linestyles='dashdot', lw=1)
plt.ylim((0, 0.5))
plt.xlim((-6, 6))
plt.ylabel('Density')
plt.xlabel('Eigenvalue')
plt.title('Spectrum of $A$')
plt.savefig("../figures/bipartite/spectrum_n_%d_k1_%d_k2_%d.png"
            % (num_vertex, deg[(0,1)], deg[(1,0)]),
                dpi=200)

plt.figure(1)
plt.hist(eig_arr)
#l_2_guess = 2*np.power((deg[(0,1)]-1)*(deg[(1,0)]-1), 0.25)
plt.vlines(l_2_guess, 0, plt.ylim()[1],
               colors='r', linestyles='dashed', lw=2)
print("|l_2|: emp = %f, guess = %f" %(l_2,l_2_guess))
plt.xlabel('Second eigenvalue $\lambda_2(A)$')
plt.ylabel('Frequency')
plt.savefig("../figures/bipartite/l_2_n_%d_k1_%d_k2_%d.png"
            % (num_vertex, deg[(0,1)], deg[(1,0)]),
                dpi=200)

## Nonbacktracking spectrum
B_eigs = compute_spectrum(gsamp, matrix="nonbacktracking")

plt.figure(2)
plt.scatter(B_eigs.real, B_eigs.imag, alpha = 0.15)
plt.xlim((-4,4))
plt.ylim((-4,4))
plt.axis('equal')
plt.gca().add_artist(plt.Circle((0,0),
    np.power((deg[(0,1)] - 1) * (deg[(1,0)] - 1), 0.25),
    color='k', linestyle='dashed', fill=False))
plt.ylabel('Imaginary part')
plt.xlabel('Real part')
plt.title('Spectrum of $B$')
plt.savefig("../figures/bipartite/spectrum_B_n_%d_k1_%d_k2_%d.png"
            % (num_vertex, deg[(0,1)], deg[(1,0)]),
                dpi=200)
