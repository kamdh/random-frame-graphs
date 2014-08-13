import numpy as np
import scipy

kavg = 5

def _sample_degree():
    return np.random.poisson(lam=kavg)

def _sample_excess_degree():
    return np.random.poisson(lam=kavg)

def eval_frac_y(z, maxiter=600, verbose=False):
    r = np.zeros((maxiter,), dtype=np.complex)
    a = 0.
    b = -1.
    P_older = 1.
    Q_older = 0.
    P_old = a
    Q_old = 1.
    for j in range(maxiter):
        a = z
        b = _sample_excess_degree()
        P_new = a*P_old + b*P_older
        Q_new = a*Q_old + b*Q_older
        P_older = P_old
        Q_older = Q_old
        P_old = P_new
        Q_old = Q_new
        r[j] = P_new/Q_new
        if verbose:
            print r[j]
    if verbose:
        return r
    else:
        return r[j]

eval_frac_v = np.vectorize(eval_frac_y)
