# misc
# submodule for miscellaneous function for
# tools subpackage

def set_bound_tau(tau):
    '''
    Setting bound for lifetime constant

    Args:
      tau: lifetime constant

    Returns:
     list of upper bound and lower bound of tau
    '''
    bound = [tau/2, 1]
    if 0.1 < tau <= 10:
        bound = [0.05, 100]
    elif 10 < tau <= 100:
        bound = [5, 500]
    elif 100 < tau <= 1000:
        bound = [50, 2000]
    elif 1000 < tau <= 5000:
        bound = [500, 10000]
    elif 5000 < tau <= 50000:
        bound = [2500, 100000]
    elif 50000 < tau <= 500000:
        bound = [25000, 1000000]
    elif 500000 < tau <= 1000000:
        bound = [250000, 2000000]
    elif 1000000 < tau:
        bound = [tau/4, 4*tau]
    return bound