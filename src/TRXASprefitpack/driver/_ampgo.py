'''
_ampgo:
Adaptive Memory Programing For Global Optimization
Based On Andrea Gavana's Implementation (see: http://infinity77.net/global_optimization/)

For implementation detail see the `Tabu Tunneling Method` section in the below paper.
L. Lasdon et al. Computers & Operations Research 37 (2010) 1500–1509
http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

:copyright: 2021-2022 by pistack (Junho Lee).
'''

from typing import Callable, Optional, Union
import numpy as np
from scipy.optimize import minimize, OptimizeResult

SCIPY_LOCAL_SOLVER = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']


def ampgo(fun: Callable, x0: np.ndarray,
          tot_iter: Optional[int] = 20, max_tunnel: Optional[int] = 5,
          tol_tunnel: Optional[float] = 1e-5,
          minimizer_kwargs: Optional[dict] = None,
          eps1: Optional[float] = 0.02, eps2: Optional[float] = 0.1,
          n_tabu: Optional[int] = 5, strategy: Optional[str] = 'farthest',
          seed: Optional[Union[int, np.random.RandomState]] = None,
          callback: Optional[Callable] = None,
          disp: Optional[bool] = False) -> OptimizeResult:
    '''
    ampgo: Adaptive Memory Programming for Global Optimization
     Based on Tabu Tunneling Method

    Args:
     fun: Objective function.
      Objective function should have following form `f(x, *args)`
     x0: initial guess
     tot_iter: maximum number of global iteration
     max_tunnel: maximum number of tunneling phase
     tol_tunnel: Tolerance to determine whether tunneling phase is
      successful or not.
      If :math:`f(x_{best}) \\geq 0` and :math:`f(x_{tunnel}) < (1+{tol})f(x_{best})+{tol}` or
      :math:`f(x_{best}) < 0` and :math:`f(x_{tunnel}) < (1-{tol})f(x_{best})+{tol}` then
      such tunneling phase is regarded as successful phase.
     minimizer_kwargs: Extra keyword arguments to be passed to the local minimizer
      `scipy.optimize.minimize`. Some important options could be:

        * method (str): The minimization Method (default: `L-BFGS-B`)
        * args (tuple): The extra arguments passed to the objective function (`fun`) and its derivatives (`jac`, `hess`)
        * jac: jacobian of objective function (see scipy.optimize.minimize)
        * hess: hessian of objective function (see scipy.optimize.minimize)
        * bounds (Sequence of Tuple): Boundary of variable (see scipy.optimize.minimize)

     eps1: Constant used to define aspiration value
     eps2: Perturbation factor used to move away from the latest local minimum
     n_tabu: size of tabulist
     strategy ({'farthest', 'oldest'}): The strategy to delete element of tabulist when
      the size of tabulist exceeds `n_tabu`.

      * `farthest`: Delete farthest point from the latest local minimum point
      * `oldest`: Delete oldest point
     seed: seed used for random perturbation of local minimum. This argument is useful when
      someone wants to reproduce optimization results.
     callback: callback function to monitor each global iteration and tunneling phase.
      function signature should be `callback(x, f_val)`
     disp: display level, If zero or `None`, then no output is printed on screen.
      If postive number then status messages are printed.


    Returns:
     The optimization results represented as a `scipy.OptimizeResult` object.
     The important attributes are

     * `x`: The solution of the optimization
     * `fun`: values of objective fuction.
     * `success`: Whether or not the optimizer exited successfuly
     * `message`: Description of the cause of the termination

    Note:
     The implementation of ampgo method is based on
     L. Lasdon et al. Computers & Operations Research 37 (2010) 1500–1509. and
     Andrea Gavana's Python Implementation.
    '''

    # initialize
    nfev = 0
    total_tunnel = 0
    success_tunnel = 0
    f_best = np.inf
    tabulist = []
    tabu_size = 0
    x0 = np.atleast_1d(x0)
    n_param = x0.size
    ub = np.empty(n_param)
    lb = np.empty(n_param)

    monitor = False

    if isinstance(seed, np.random.RandomState):
        rng = seed
    elif isinstance(seed, int):
        rng = np.random.RandomState(seed)
    elif seed is None:
        rng = np.random.RandomState(None)
    else:
        raise Exception('Invalid seed for random generator')

    if callable(callback):
        monitor = True

    # Setting local minimizer
    if minimizer_kwargs is None:
        method = 'L-BFGS-B'
        args = ()
        bounds = None
        jac = None
        hess = None
        hessp = None
        tol = 1e-8
        minimizer_kwargs = {}
    else:
        method = minimizer_kwargs.pop('method', 'L-BFGS-B')
        args = minimizer_kwargs.pop('args', ())
        bounds = minimizer_kwargs.pop('bounds', None)
        jac = minimizer_kwargs.pop('jac', None)
        hess = minimizer_kwargs.pop('hess', None)
        hessp = minimizer_kwargs.pop('hessp', None)
        tol = minimizer_kwargs.pop('tol', 1e-8)

    if method not in SCIPY_LOCAL_SOLVER:
        raise Exception('Invalid local solver, local solver should be one of' +
        '[' + ', '.join(SCIPY_LOCAL_SOLVER) + ']')

    # Setting Arguments For tabu tunneling Function
    if callable(jac):
        ttf = wrapper_tunnel(fun, *args)
        jac_ttf = wrapper_grad_tunnel(fun, jac, *args)
    elif jac is True:
        ttf = wrapper_fun_grad_tunnel(fun, *args)
        jac_ttf = True
    else:
        ttf = wrapper_tunnel(fun, *args)
        jac_ttf = None

    if bounds is None:
        for i in range(n_param):
            ub[i] = np.inf
            lb[i] = -np.inf
    else:
        if len(bounds) != n_param:
            raise Exception(
                'Length of Bounds and the number of paramter should be same')
        for i in range(n_param):
            if bounds[i] is None:
                ub[i] = np.inf
                lb[i] = -np.inf
            else:
                lb[i], ub[i] = bounds[i]
                if lb[i] is None:
                    lb[i] = -np.inf
                if ub[i] is None:
                    ub[i] = np.inf

    # Start main loop
    for i in range(tot_iter+1):
        res = minimize(fun, x0, args=args, method=method,
                       jac=jac, hess=hess, hessp=hessp, bounds=bounds, tol=tol,
                       **minimizer_kwargs)
        f_opt = res['fun']
        x0 = res['x']
        nfev = nfev + res['nfev']
        if monitor:
            callable(x0, f_opt)
        if disp:
            print('='*72)
            print(f'local minimum is found in global iteration: {i}')
            if not res['success']:
                print('Warning: local optimization is failed')
        if f_opt < f_best:
            if disp:
                print(
                    f'local minimum improves solution at global iteration: {i}')
                print(f'Current: {f_opt} | previous: {f_best}')
            f_best = f_opt
            x_best = x0

        # If size of tabulist exceeds n_tabu then delete element
        # Add local minimum solution to the tabulist

        if tabu_size > n_tabu-1:
            tabulist = delete_element(x0, tabulist, strategy)
            tabu_size = n_tabu-1
        tabulist.append(x0)
        tabu_size = tabu_size+1

        # Calculates Aspiration
        aspiration = f_best - eps1*(1+np.abs(f_best))

        success = False
        for j in range(max_tunnel):
            total_tunnel = total_tunnel+1
            if disp:
                print('='*72)
                print(f'Tunneling Phase is Started: {i}-{j+1}')

            # Perturbe local minimum point x0
            vaild = False
            while not vaild:
                r = rng.uniform(-1, 1, n_param)
                beta = (eps2*(1+np.linalg.norm(x0)))/np.linalg.norm(r)
                x_try = x0 + beta*r
                x_try = np.where(x_try < lb, lb, x_try)
                x_try = np.where(x_try > ub, ub, x_try)
                vaild = check_vaild(x_try, tabulist)

            # start tabu tunneling
            res = minimize(ttf, x_try, args=(aspiration, tabulist), method=method,
                           jac=jac_ttf, bounds=bounds, tol=tol, **minimizer_kwargs)
            x0 = res['x']
            nfev = nfev + res['nfev']
            if jac is True:
                f_opt, _ = fun(x0, *args)
            else:
                f_opt = fun(x0, *args)
            nfev = nfev+1

            if f_best >= 0 and f_opt < (1+tol_tunnel)*f_best+tol_tunnel:
                success = True
                success_tunnel = success_tunnel+1
            elif f_best < 0 and f_opt < (1-tol_tunnel)*f_best+tol_tunnel:
                success = True
                success_tunnel = success_tunnel+1

            if monitor:
                callback(x0, f_opt)

            if disp and not res['success']:
                print('Warning: local optimization is failed')

            if disp and success:
                print('Tunneling phase is successful')

            if f_opt < f_best:
                if disp:
                    print(f'Tunneling phase {i}-{j+1} improves solution')
                    print(f'Current: {f_opt} | previous: {f_best}')
                f_best = f_opt
                x_best = x0

            # update tabulist
            if tabu_size > n_tabu-1:
                delete_element(x0, tabulist, strategy)
                tabu_size = n_tabu-1
            tabulist.append(x0)
            tabu_size = tabu_size+1

            if success:
                break

    # Optimization Process is finished

    result = OptimizeResult()
    result['fun'] = f_best
    result['x'] = x_best
    result['nfev'] = nfev
    result['success'] = True
    result['message'] = \
        ['Requested Number of global iteration is finished.']

    return result


def check_vaild(x_try, tabulist):
    '''
    Check random pertubation of latest local mimum point
    is vaild.
    '''
    dist = np.sum((tabulist-x_try)**2, axis=1)
    min_dist = np.min(dist)
    return min_dist > 1e-16


def delete_element(x_local, tabulist, strategy):
    '''
    Delete element from tabulist
    '''
    if strategy == 'oldest':
        tabulist.pop(0)
    else:
        dist = np.sum((tabulist-x_local)**2, axis=1)
        idx = np.argmax(dist)
        tabulist.pop(idx)

    return tabulist

def wrapper_tunnel(fun, *fun_args):
    '''
    wrapper function for tabu tunneling function
    '''
    def tunnel(x0, aspiration, tabulist):
        numerator = (fun(x0, *fun_args)-aspiration)**2
        denumerator = 1
        for tabu in tabulist:
            denumerator = denumerator*np.sum((x0-tabu)**2)
        return numerator/denumerator
    return tunnel

def wrapper_grad_tunnel(fun, jac, *fun_args):
    '''
    wrapper function for gradient of tabu tunneling function
    '''
    def grad_tunnel(x0, aspiration, tabulist):
            fval = fun(x0, *fun_args) - aspiration
            numerator = fval**2
            grad_numerator = fval*jac(x0, *fun_args)
            denominator = 1
            grad_denom = np.zeros_like(x0)
            for tabu in tabulist:
                diff = tabu-x0
                dist = np.sum(diff**2)
                denominator = denominator*dist
                grad_denom = grad_denom + diff/dist
            return 2*(grad_numerator+numerator*grad_denom)/denominator
    return grad_tunnel

def wrapper_fun_grad_tunnel(fun, *fun_args):
    '''
    wrapper function for pair of tabu tunneling function and
    its gradient
    '''
    def fun_grad_tunnel(x0, aspiration, tabulist):
        f_val, grad_val = fun(x0, *fun_args)
        f_val = f_val-aspiration
        numerator = f_val**2
        grad_numerator = f_val*grad_val
        denominator = 1
        grad_denominator = np.zeros_like(x0)
        for tabu in tabulist:
            diff = tabu-x0
            dist = np.sum(diff**2)
            denominator = denominator*dist
            grad_denominator = grad_denominator + \
                diff/dist
        
        y_ttf = numerator/denominator
        deriv_y_ttf = 2*(grad_numerator/denominator +
        y_ttf*grad_denominator)
        return y_ttf, deriv_y_ttf
    
    return fun_grad_tunnel
