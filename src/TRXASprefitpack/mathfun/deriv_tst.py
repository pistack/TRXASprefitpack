'''
deriv_tst:
submodule to test derivative routine of mathfun subpackage

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

import numpy as np

def test_num_deriv(func: callable, *args, eps=1.4901161193847656e-08):
    '''
    Test implementation of derivative via finite difference
    '''
    n = len(args)
    if isinstance(args[0], np.ndarray):
        num_grad = np.empty((n, args[0].size))
        for i in range(n):
            f_args = list(args)
            b_args = list(args)
            f_args[i] = f_args[i]+eps
            b_args[i] = b_args[i]-eps
            f_args = tuple(f_args)
            b_args = tuple(b_args)
            num_grad[i, :] = (func(*f_args)-func(*b_args))/(2*eps)
    else:
        num_grad = np.empty(n)
        for i in range(n):
            f_args = list(args)
            b_args = list(args)
            f_args[i] = f_args[i]+eps
            b_args[i] = b_args[i]-eps
            f_args = tuple(f_args)
            b_args = tuple(b_args)
            num_grad[i] = (func(*f_args)-func(*b_args))/(2*eps)
    
    return num_grad