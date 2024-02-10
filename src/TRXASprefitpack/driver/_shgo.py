'''
_shgo:
Wrapper for scipy.optimize.shgo global optimization method

:copyright: 2021-2024 by pistack (Junho Lee).
'''

from scipy.optimize import shgo


def _wrapper_shgo(func, x0, constraints=None,
                  n=None, iters=1, callback=None,
                  minimizer_kwargs=None, options=None,
                  sampling_method='simplicial'):
    
    bounds = minimizer_kwargs.pop('bounds', None)
    args = minimizer_kwargs.pop('args', None)
    
    return shgo(func, bounds, args=args,
                constraints=constraints, n=n,
                iters=iters, callback=callback,
                minimizer_kwargs=minimizer_kwargs, options=options,
                sampling_method=sampling_method)