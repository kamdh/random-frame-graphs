from frame import Frame
import networkx as nx

if __name__=="main":
    ps = {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2}
    # # bad
    # deg = {(0,1): 3, (1,2):6, (2,3):2, 
    #        (1,0): 2, (2,1):6, (3,2): 10}
    # good
    deg = {(0,1): 3, (1,2):6, (2,3):2, 
           (1,0): 2, (2,1):6, (3,2):3}
    g = Frame(nx.DiGraph(nx.path_graph(4)), p=ps, deg=deg)
