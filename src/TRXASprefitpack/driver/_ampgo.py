# _ampgo.py
# light-wighten ampgo routine originated from Andrea Gavana (MIT licence)
# Modification
# 1. supports user supplied gradient function
# 2. drop supports for openopt local solver
# 3. Change type of return value from tuple to scipy.optimize.OptimizeResult class
# 4. Add epsilon to denumerator to avoid divide by zero error
# Copyright 2014 Andrea Gavana
# Copytight 2022 Junho Lee (pistack)

from typing import Callable, Optional
import numpy as np
from scipy.optimize import minimize, OptimizeResult


def ampgo(func: Callable, x0: np.ndarray, maxfev: Optional[int] = None, totaliter: Optional[int] = 20,
          maxiter: Optional[int] = 5, glbtol: Optional[float] = 1e-5,
          minimizer_kwargs: Optional[dict] = None, 
          eps1: Optional[float] = 0.02, eps2: Optional[int] = 0.1, tabulistsize: Optional[int] = 5,
          tabustrategy: Optional[str]='farthest', fmin: Optional[float] = -np.inf, disp: Optional[int] = None) -> OptimizeResult:
    """
    Finds the global minimum of a function using the AMPGO (Adaptive Memory Programming for
    Global Optimization) algorithm. 

    Args:
     func: Function to be optimized, which has the form ``f(x, *args)``.
     x0: initial guess
     maxfev: The maximum number of total function evaluations allowed. [default: 100*n_parm]
     totaliter:  The maximum number of global iterations allowed.
     maxiter: The maximum number of `Tabu Tunnelling` iterations allowed during each global iteration.     
     glbtol: The optimization will stops when `best_f^{k+1} < best_f^{k}(1+glbtol)`.
             Where `best_f^k` is the global minimum of objective function for kth global iteration.
     minimizer_kwargs: Extra keyward arguments to be passed to the local solver ``scipy.optimize.minimize()`.
                       Some important options could be

                          method: Type of solver [defalut: `L-BFGS-B`]
                          args: the arguments passed to object function(`func`) and its derivatives(`jac`, `hess`)
                          bounds: Bounds on variable for both global and local solvers
     eps1: A constant used to define an asipartion value for the objective function during the Tunnelling phase.
     eps2: Perturbation factor used to move away from the latest local minimum at the start of a Tunnelling phase.
     tabulistsize: The size of the tabu search list
     tabustrategy: The strategy to use when the size of the tabu list exceeds `tabulistsize`.
                   Available strategy [default: `farthest`]
                    
                     `oldest`: Drop the oldest point from the tabu list
                     `farthest`: Drop the element farthest from the last local minimum found.
     fmin: If known, the global minimum of objective function.
     disp: display level, If zero or `None`, then no output is printed on scrren. If a positive number, then status
           messages are printed.
     
     Returns:
      The optimization results represented as a `scipy.OptimizeResult` object.
      The important attributes are
       `x`: The solution of the optimization
       `success`: Whether or not the optimizer exited successfuly
       `message`: Description of the cause of the termination
       `all_tunnel`: The number of total tunneling phase performed
       `success_tunnel`: The number of successful tunneling phase performed

     Note:
     The detailed implementation of AMPGO is described in the paper 
     "Adaptive Memory Programming for Constrained Global Optimization" located here:
     http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf
     Copyright 2014 Andrea Gavana (Original code)
     Copyright 2022 Junho Lee(pistack) (Modification)
    """

    method = minimizer_kwargs.pop('method', 'L-BFGS-B')
    args = minimizer_kwargs.pop('args', ())
    bounds = minimizer_kwargs.pop('bounds', None)
    jac = minimizer_kwargs.pop('jac', None)
    hess = minimizer_kwargs.pop('hess', None)
    hessp = minimizer_kwargs.pop('hesp', None)
    tol = minimizer_kwargs.pop('tol', 1e-8)

    global_res = OptimizeResult()
        
    x0 = np.atleast_1d(x0)
    n = len(x0)

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    low = np.empty(n, dtype=float)
    up = np.empty(n, dtype=float)

    for i in range(n):
        if bounds[i] is None:
            l, u = -np.inf, np.inf
        else:
            l, u = bounds[i]
            if l is None:
                low[i] = -np.inf
            else:
                low[i] = l
            if u is None:
                up[i] = np.inf
            else:
                up[i] = u

    if maxfev is None:
        maxfev = 1000*len(x0)

    if tabulistsize < 1:
        raise Exception(f'Invalid tabulistsize specified: {tabulistsize}. It should be an integer greater than zero.')
    if tabustrategy not in ['oldest', 'farthest']:
        raise Exception(f'Invalid tabustrategy specified: {tabustrategy}. It must be one of "oldest" or "farthest"')

    if disp is None or disp <= 0:
        disp = 0

    tabulist = []
    best_f = np.inf
    best_x = x0
    
    global_iter = 0
    all_tunnel = success_tunnel = 0
    evaluations = 0

    if glbtol < tol:
        tol = glbtol
    
    terminate = False

    while not terminate:

        if disp > 0:
            print('\n')
            print('='*72)
            print('Starting MINIMIZATION Phase %-3d'%(global_iter+1))
            print('='*72)

        options = {'maxiter': max(1, maxfev), 'disp': disp}
        if minimizer_kwargs is not None:
            options.update(minimizer_kwargs)
            res = minimize(func, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, tol=tol, options=options)
            xf, yf, num_fun = res['x'], res['fun'], res['nfev']
        
        maxfev -= num_fun
        evaluations += num_fun

        if yf < best_f:
            best_f = yf
            best_x = xf

        if disp > 0:
            print('\n\n ==> Reached local minimum: %s\n'%yf)
        
        if best_f < fmin*(1+glbtol):
            if disp > 0:
                print('='*72)
                terminate = True
                msg = 'Optimization terminated successfully'
                status = 0
                success = True

        if maxfev <= 0:
            if disp > 0:
                print('='*72)
                terminate = True
                msg = 'Maximum number of function evaluations exceeded'
                status = -1
                success = False

        tabulist = drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy)
        tabulist.append(xf)

        i = improve = 0

        while i < maxiter and improve == 0 and not terminate:

            if disp > 0:
                print('-'*72)
                print(f'Starting TUNNELLING   Phase ({global_iter+1:3d}-{i+1:3d})')
                print('-'*72)

            all_tunnel += 1
            
            r = np.random.uniform(-1.0, 1.0, size=n)
            beta = eps2*np.linalg.norm(xf)/np.linalg.norm(r)
            
            if np.abs(beta) < 1e-8:
                beta = eps2
                
            x0  = xf + beta*r

            x0 = np.where(x0 < low, low, x0)
            x0 = np.where(x0 > up , up , x0)

            aspiration = best_f - eps1*(1.0 + np.abs(best_f))

            tunnel_args = tuple([func, jac, aspiration, tabulist] + list(args))

            options = {'maxiter': max(1, maxfev), 'disp': disp}
            if minimizer_kwargs is not None:
                options.update(minimizer_kwargs)
            
            tunnel_jac = None
            if jac is not None:
                tunnel_jac = grad_tunnel 
            
            res = minimize(tunnel, x0, args=tunnel_args, method=method, jac=tunnel_jac, bounds=bounds, tol=tol, options=options)
            xf, yf, num_fun = res['x'], res['fun'], res['nfev']

            maxfev -= num_fun
            evaluations += num_fun

            yf = inverse_tunnel(xf, yf, aspiration, tabulist)

            if yf <= best_f*(1+glbtol):
                oldf = best_f
                best_f = yf
                best_x = xf
                improve = 1
                success_tunnel += 1

                if disp > 0:
                    print(f'\n\n ==> Successful tunnelling phase. Reached local minimum: {yf} < {oldf}\n')

                if best_f < fmin*(1+glbtol):
                    terminate = True
                    msg = 'Optimization terminated successfully'
                    status = 0
                    success = True
                
                elif maxfev <= 0:
                    terminate = True
                    msg = 'Maximum number of function evaluations exceeded'
                    status = -1
                    success = False

            i += 1                   

            tabulist = drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy)
            tabulist.append(xf)

        if disp > 0:
            print('='*72)

        global_iter += 1
        x0 = xf.copy()

        if global_iter >= totaliter:
            terminate = True
            msg = 'Maximum number of global iterations exceeded'
            status = -1
            success = False

        if best_f < fmin + glbtol:
            terminate = True
            msg = 'Optimization terminated successfully'
            status = 0
            success = True
        
    global_res['x'] = best_x
    global_res['fun'] = best_f
    global_res['success'] = success
    global_res['status'] = status
    global_res['message'] = msg
    global_res['nfev'] = evaluations
    global_res['nit'] = global_iter
    global_res['all_tunnel'] = all_tunnel
    global_res['success_tunnel'] = success_tunnel
    
    return global_res


def drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy):
    '''
    Remove tabu points from tabulist when size of tabulist exceeded `tabulistsize`.
    '''

    if len(tabulist) < tabulistsize:
        return tabulist
    
    if tabustrategy == 'oldest':
        tabulist.pop(0)
    else:
        distance = np.sqrt(np.sum((tabulist-xf)**2, axis=1))
        index = np.argmax(distance)
        tabulist.pop(index)

    return tabulist


def tunnel(x0, *args):
    '''
    Tunneling function
    '''
    
    func, jac, aspiration, tabulist = args[0:4]

    fun_args = ()    
    if len(args) > 4:
        fun_args = tuple(args[4:])

    numerator = (func(x0, *fun_args) - aspiration)**2
    denominator = 1.0

    for tabu in tabulist:
        denominator = denominator*(np.linalg.norm(x0 - tabu)+1e-8)

    ytf = numerator/denominator

    return ytf

def grad_tunnel(x0, *args):
    '''
    gradient of tunneling function
    '''
    func, jac, aspiration, tabulist = args[0:4]

    fun_args = ()
    if len(args) > 4:
        fun_args = tuple(args[4:])
    
    fval = func(x0, *fun_args) - aspiration
    numerator = fval**2
    grad_numerator = 2*fval*jac(x0, *fun_args)
    denominator = 1.0
    grad_denom = np.zeros_like(x0)

    for tabu in tabulist:
        diff = tabu-x0
        dist = np.linalg.norm(diff)
        denominator = denominator*(1e-8+dist)
        grad_denom = grad_denom + diff/(1e-16+dist**2)
    
    gradtf = (grad_numerator+numerator*grad_denom)/denominator

    return gradtf


def inverse_tunnel(xtf, ytf, aspiration, tabulist):
    '''
    Estimate value of object function when value of tunneling function is given.
    '''

    denominator = 1.0

    for tabu in tabulist:
        denominator = denominator*(1e-8+np.linalg.norm(xtf - tabu))
    
    yf = aspiration + np.sqrt(ytf*denominator)
    return yf