import numpy as np
from fractions import gcd

def lambda2(d1, d2):
    """Return second eigenvalue"""
    return np.sqrt(d1 - 1) + np.sqrt(d2 - 1)

def lcm(*numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) / gcd(a, b)
    return reduce(lcm, numbers, 1)

def dmin_JL(d1, d2, code1, code2):
    # Janwa & Lal, 2002
    dmin = 1./(d1*d2) * (code1[2]*code2[2] - lambda2(d1, d2)/2.*(code1[2]+code2[2]))
    return dmin

def check_codes(code1, code2):
    d1 = code1[0]
    d2 = code2[0]
    rate1 = float(code1[1]) / code1[0]
    rate2 = float(code2[1]) / code2[0]
    rate = rate1 + rate2 - 1.0
    dmin = dmin_JL(d1, d2, code1, code2)
    min_graph_n = np.ceil(2./(dmin * d1)).astype(int)
    lcm_degs = lcm(d1, d2)
    if min_graph_n * d1 < lcm_degs:
        min_graph_n = lcm_degs / d1
    else:
        frac_min = np.ceil( min_graph_n / (lcm_degs / d1) )
        min_graph_n = frac_min * lcm_degs / d1
    if dmin > 0 and rate > 0 and \
      code1[2] >= code2[2] and 2*code2[2] > lambda2(d1, d2):
        print "\nTotal degree: %d " % d_total
        print "found code with rate = %f and distance = %f" % (rate, dmin)
        print "min graph size for distance 2: n = %d, m = %1.1f" % \
          (min_graph_n, min_graph_n * d1 / float(d2))
        print code1
        print code2
        return True
    else:
        if dmin <= 0:
            print "dmin <= 0"
        if rate <= 0:
            print "rate <= 0"
        if code1[2] < code2[2]:
            print "code 1 has distance < code 2"
        if 2*code2[2] <= lambda2(d1, d2):
            print "distance bound on 2nd eig not met"
        return False


# def dmin_HJ(d1, d2, code1, code2):
#     # Hoholdt & Justesen, 2011
#     from scipy.optimize import newton
#     alpha = 2.
#     dist1 = code1[2]
#     dist2 = code2[2]
#     beta = newton(lambda x: x**2 * (alpha * d_1

rate_factor = 0

d_total = 4
while d_total <= 128:
    for d2 in np.arange(2, np.floor(d_total/2.)+1):
        for rate_factor_1 in range(-6,7):
            for rate_factor_2 in range(-6,7):
                d1 = d_total - d2
                code1 = (d1,
                        np.floor(d1/2.) + 1 + rate_factor_1,
                         d1 - np.floor(d1/2.) - rate_factor_1)
                code2 = (d2,
                         np.floor(d2/2.) + 1 + rate_factor_2,
                         d2 - np.floor(d2/2.) - rate_factor_2)
                rate1 = float(code1[1]) / code1[0]
                rate2 = float(code2[1]) / code2[0]
                rate = rate1 + rate2 - 1.0
                dmin = dmin_JL(d1, d2, code1, code2)
                min_graph_n = np.ceil(2./(dmin * d1)).astype(int)
                lcm_degs = lcm(d1, d2)
                if min_graph_n * d1 < lcm_degs:
                    min_graph_n = lcm_degs / d1
                else:
                    frac_min = np.ceil( min_graph_n / (lcm_degs / d1) )
                    min_graph_n = frac_min * lcm_degs / d1
                if dmin > 0 and rate > 0 and \
                  code1[2] >= code2[2] and 2*code2[2] > lambda2(d1, d2):
                    print "\nTotal degree: %d " % d_total
                    print "found code with rate = %f and distance = %f" % \
                      (rate, dmin)
                    print "min graph size for distance 2: n = %d, m = %1.1f" % \
                      (min_graph_n, min_graph_n * d1 / float(d2))
                    print code1
                    print code2
    d_total += 1

