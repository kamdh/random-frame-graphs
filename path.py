from frame import Frame
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

epsilon=0.00001
offset=0.5

if __name__=="__main__":
    print "entering main"
    #ps = {0: 0.1, 1: 0.5, 2: 0.3, 3: 0.05, 4: 0.05}
    ps = {0: 0.05, 1: 0.25, 2: 0.4, 3: 0.25, 4: 0.05}
    # # bad
    # deg = {(0,1): 3, (1,2):6, (2,3):2, 
    #        (1,0): 2, (2,1):6, (3,2): 10}
    # good
    #deg = {(0,1): 5, (1,2):6, (2,3):2, (3,4): 2,
    #       (1,0): 1, (2,1):10, (3,2):12, (4,3):2}
    deg = {(0,1): 50, (1,2):40, (2,3):25, (3,4):10,
           (1,0): 10, (2,1):25, (3,2):40, (4,3):50}
    xs = np.linspace(-2,2,701)
    g = Frame(nx.DiGraph(nx.path_graph(5)), p=ps, deg=deg)
    Aspec = g.spectrum(xs, epsilon=epsilon, offset=offset)
    plt.ion()
    plt.figure()
    plt.plot(xs, np.abs(Aspec), 'b-', linewidth=2)
    raw_input()
