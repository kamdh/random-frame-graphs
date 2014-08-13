#!/usr/bin/env python
from spectrum import compute_spectrum
import tripartite as tri
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from frame import Frame
from IPython import embed
import networkx as nx

'''
Generate a random regular tripartite graph.
Then, compute its eigenspectrum and plot the empirical spectral density.
'''
def main():
    n=3600
    alpha=0
    simple = True
    epsilon =0.001
    xs = np.linspace(-6,6,401)
    pA = 0.125
    pB = 0.25
    kAB = 2
    kAC = 10
    kBC = 5
    # pA = 0.125
    # pB = 0.125
    # kAB = 3
    # kAC = 6
    # kBC = 12
    # pA = 0.125
    # pB = 0.25
    # kAB = 2
    # kAC = 0
    # kBC = 5
    ## derived parameters:
    pC = 1-pA-pB
    kBA = (pA/pB)*kAB
    kCB = (pB/pC)*kBC
    kCA = (pA/pC)*kAC
    print "Generating graphs"
    g = tri.tripartite_regular_configuration_graph(n, pA, pB, kAB, kAC, kBC,
                                                   verbose=True) 
    print "Diagonalizing matrix"
    #embed()    
    L = compute_spectrum(g)
    print "Plotting"
    plt.ion()
    plt.figure(0)
    plt.hist(L, bins=301, normed=True)
    print "Computing spectrum analytically"
    print "  1. tripartite specific code"
    Lanalytic = tri.tripartite_regular_spectrum(xs, pA, pB, kAB, kAC, kBC,
                                                epsilon=epsilon)
    print "  2. general Frame code"
    triFrame = Frame(nx.complete_graph(3, create_using=nx.DiGraph()),
                     p={0:pA, 1:pB, 2:1-pA-pB},
                     deg={(0,1):kAB, (0,2):kAC, (1,0):kBA, (1,2):kBC,
                          (2,0):kCA, (2,1):kCB})
    Lanalytic2 = triFrame.spectrum(xs)
    plt.hold(True)
    plt.plot(xs, Lanalytic, 'b-', linewidth=2)
    plt.plot(xs, Lanalytic2, 'r-', linewidth=2)
    plt.ylim((0, 0.6))
    # A = tri.block_gaussian_guess(n, pA, pB)
    # Lguess = eigvalsh(A, check_finite=False)
    # Lguess /= np.sqrt(n)
    # plt.hist(Lguess, bins=301, normed=True)
    raw_input()
    embed()

if __name__ == '__main__':
    main()
