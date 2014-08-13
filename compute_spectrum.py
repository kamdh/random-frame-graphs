#!/usr/bin/env python
from spectrum import *


'''
Generate a graph with a given, simple degree distribution via the configuration
model. Then, compute its eigenspectrum and plot the empirical spectral density.
'''

ps = np.array( [0.5, 0.5] )
ks = np.array( [5, 2] ) # degrees
n=2000
alpha=0
simple = True

print "Generating graphs"
G = gen_delta_graph(n, ks, ps, simple, model='conf')
Gcl = gen_delta_graph(n, ks, ps, simple, model='chung-lu')
print "Computing spectra"
L = compute_spectrum(G)
Lcl = compute_spectrum(Gcl)

print "Plotting"
plt.ion()
plt.figure(0)
plt.hist(L, bins=301, normed=True)
plt.figure(1)
plt.hist(Lcl, bins=301, normed=True)
plt.show()

print "Resolvent stuff"
xs = np.linspace(-5,5,21)
epsilon = 0.05
print "Monte Carlo"
mu1 = spectrum_mc(xs, epsilon, ks, ps, alpha)
print "Mean-field"
mu2 = spectrum_analytic(xs, epsilon, ks, ps, alpha)
for i in range(2):
    plt.figure(i)
    plt.hold(True)
    plt.plot(xs, np.real(mu1), 'b-')
    plt.plot(xs, np.real(mu2), 'r--')
