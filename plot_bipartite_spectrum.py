from framegraph.spectrum import compute_spectrum
from framegraph.frame import Frame
from framegraph.nonbacktracking import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def MP_law(x, d1, d2):
    from Dumitriu & Johnson
    dR = max(d1, d2)
    dL = min(d1, d2)
    y = float(dR) / dL
    a = 1 - np.sqrt(y)
    b = 1 + np.sqrt(y)
    return 0.5 * y / (np.pi * np.abs(x)) * \
        np.sqrt((b**2 - np.abs(x)) * (np.abs(x) - a**2))
    return y / ((1 + y) * np.pi * np.abs(x)) * \
      np.sqrt((b**2 - x**2) * (x**2 - a**2))

def bipartite_law(x, d1, d2):
    # from Godsil & Mohar (1988), Corollary 4.5
    p = np.sqrt((d1 - 1) * (d2 - 1))
    return d1 * d2 / (np.pi * (d1 + d2) * (d1 * d2 - x ** 2) * np.abs(x)) \
      * np.sqrt((- x ** 2 + d1 * d2 - (p - 1) ** 2) * \
                    (x ** 2 - d1 * d2 + (p + 1) ** 2))

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.close('all')

epsilon = 0.00001
offset = 0.5
num_vertex = 120
num_rep = 20

ps = {0: 0.7, 1: 0.3}
deg = {(0,1): 3, (1,0): 7}
g = Frame(nx.DiGraph(nx.path_graph(2)), p=ps, deg=deg)

## Particular analytical eigenvalues
# l_2_guess = np.sqrt( 2 * np.sqrt( (deg[(0,1)] - 1) * (deg[(1,0)] - 1) )
#                     + deg[(0,1)] + deg[(1,0)] - 2)
l_2_guess = np.sqrt(deg[(0,1)] - 1) + np.sqrt(deg[(1,0)] - 1)
l_1 = np.sqrt( deg[(0,1)] * deg[(1,0)] )
bulk_min = - np.sqrt(deg[(0,1)] - 1) + np.sqrt(deg[(1,0)] - 1)

## Compute analytical bulk spectrum
xneg = np.linspace(-l_2_guess, -bulk_min, 100)
xpos = np.linspace(bulk_min, l_2_guess, 100)
spec_neg = g.spectrum(xneg, epsilon=epsilon, offset=offset)
spec_pos = g.spectrum(xpos, epsilon=epsilon, offset=offset)
# spec_neg = bipartite_law(xneg, deg[(0,1)], deg[(1,0)])
# spec_pos = bipartite_law(xpos, deg[(0,1)], deg[(1,0)])
# spec_neg = MP_law(xneg, deg[(0,1)], deg[(1,0)])
# spec_pos = MP_law(xpos, deg[(0,1)], deg[(1,0)])

## Test 2nd eig guess
eig_arr = []
for r in range(num_rep):
    gsamp = g.sample(num_vertex, parallel_edges=False, verbose=True)
    Aspec_samp = compute_spectrum(gsamp)
    Aspec_sort = np.sort(abs(Aspec_samp))
    l_2 = Aspec_sort[-3]
    eig_arr.append(l_2)
    print("solved rep %d, eig = %f" % (r,l_2))

while l_2 <= l_2_guess:
    # force an outlier
    gsamp = g.sample(num_vertex, parallel_edges=False, verbose=True)
    Aspec_samp = compute_spectrum(gsamp)
    Aspec_sort = np.sort(abs(Aspec_samp))
    l_2 = Aspec_sort[-3]
    eig_arr.append(l_2)
    r += 1
    print("solved rep %d, eig = %f" % (r,l_2))

eig_arr = np.array(eig_arr)

plt.ion()
plt.figure(0)
plt.plot(xneg, np.abs(spec_neg), 'k-', linewidth=1)
plt.plot(xpos, np.abs(spec_pos), 'k-', linewidth=1)
plt.hist(Aspec_samp, bins = 41, density = True)
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
plt.xlabel('Second eigenvalue $\eta$')
plt.ylabel('Frequency')
plt.savefig("../figures/bipartite/l_2_n_%d_k1_%d_k2_%d.png"
            % (num_vertex, deg[(0,1)], deg[(1,0)]),
                dpi=200)

## Nonbacktracking spectrum
print("computing non-backtracking matrix and spectrum")
B_eigs = compute_spectrum(gsamp, matrix="nonbacktracking")

plt.figure(2)
plt.scatter(B_eigs.real, B_eigs.imag, alpha = 0.15, color='orange')
plt.xlim((-4,4))
plt.ylim((-4,4))
plt.axis('equal')
plt.gca().add_artist(plt.Circle((0,0),
    np.power((deg[(0,1)] - 1) * (deg[(1,0)] - 1), 0.25),
    color='k', linestyle='dashed', fill=False))
plt.scatter([+1, -1],[0,0], marker='.', color='k')
plt.scatter([0, 0],
            [np.sqrt(deg[(0,1)] - 1), -np.sqrt(deg[(0,1)] - 1)],
             marker='*', color='b')
plt.scatter([-np.sqrt((deg[(0,1)] - 1) * (deg[(1,0)] - 1)),
             np.sqrt((deg[(0,1)] - 1) * (deg[(1,0)] - 1))],
            [0, 0],
             marker='+', color='b')
# plt.scatter([0, 0],
#             [np.sqrt(deg[(1,0)] - 1), -np.sqrt(deg[(1,0)] - 1)],
#              marker='1', color='k')
plt.axis('equal')
plt.xlim((-4,4))
plt.ylim((-4,4))
plt.ylabel('Imaginary part')
plt.xlabel('Real part')
plt.title('Spectrum of $B$')
plt.savefig("../figures/bipartite/spectrum_B_n_%d_k1_%d_k2_%d.png"
            % (num_vertex, deg[(0,1)], deg[(1,0)]),
                dpi=200)
