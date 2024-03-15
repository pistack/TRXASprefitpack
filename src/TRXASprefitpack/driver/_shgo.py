'''
_shgo:
Wrapper for scipy.optimize.shgo global optimization method

:copyright: 2021-2024 by pistack (Junho Lee).
'''

from scipy.optimize import shgo


def _wrapper_shgo(func, x0, **kwargs):


    bounds = kwargs['minimizer_kwargs'].pop('bounds', None)
    args = kwargs['minimizer_kwargs'].pop('args', None)
    sampling_method = kwargs.pop('sampling_method', 'sobol') # set default sampling method to sobol
    kwargs['sampling_method'] = sampling_method
    kwargs['args'] = args
    
    return shgo(func, bounds, **kwargs)