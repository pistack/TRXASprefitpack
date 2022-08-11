'''
_ampgo:
Adaptive Memory Programing For Global Optimization
Based On Andrea Gavana's Implementation (see: http://infinity77.net/global_optimization/)

For implementation detail see the `Tabu Tunneling Method` section in the below paper.
http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Callable, Optional
import numpy as np
from scipy.optimize import minimize, OptimizeResult

def ampgo(func: Callable, x0: np.ndarray,
          tot_iter: Optional[int] = 20, max_tunnel: Optional[int] = 5,
          tol_tunnel: Optional[float] = 1e-5,
          minimizer_kwargs: Optional[dict] = None,
          eps1: Optional[float] = 0.02, eps2: Optional[float] = 0.1,
          n_tabu: Optional[int] = 5, strategy: Optional[str] = 'farthest',
          disp: Optional[bool] = False) -> OptimizeResult:

          '''
          ampgo: Adaptive Memory Programming for Global Optimization 
           Based on Tabu Tunneling Method
          
          Args:
           func: Objective function. 
             Signiture of objective function: `f(x, *args)`
           x0: initial guess
           tot_iter: maximum number of global iteration
           max_tunnel: maximum number of tunneling phase
           tol_tunnel: Tolerance to determine whether tunneling phase is 
             successful or not. 
             If :math:`|f(x_{tunnel})}<(1+{tol})(|f(x_{best})|+{tol})` then such tunneling phase
             is regarded as successful phase.
           minimizer_kwargs: Extra keyword arguments to be passed to the local minimizer
            `scipy.optimize.minimize`. Some important options could be:
             
            * method (str): The minimization Method (default: `L-BFGS-B`)

            * args (tuple): The extra arguments passed to the objective function (`func`) and
               its derivatives (`jac`, `hess`)
             
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
           disp: display level, If zero or `None`, then no output is printed on screen.
             If postive number then status messages are printed.
          
          
          Returns:
           The optimization results represented as a `scipy.OptimizeResult` object.
           The important attributes are
           
           * `x`: The solution of the optimization
           * `fun, jac, hess`: values of objective fuction, its gradient and hessian.
           * `success`: Whether or not the optimizer exited successfuly
           * `message`: Description of the cause of the termination
          '''
          return


def check_vaild(x_try, tabulist):
    '''
    Check random pertubation of latest local mimum point
    is vaild.
    '''
    if not tabulist:
        return True
    else:
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


def ttf(x0, *args):
    '''
    Tabu Tunneling function
    '''
    func, aspiration, tabulist = args[:3]
    fun_args = ()
    if len(args)>4:
        fun_args = tuple(args[4:])
    
    numerator = (func(x0, *fun_args)-aspiration)**2
    denominator = 1
    for tabu in tabulist:
        denominator = denominator*np.linalg.norm(x0-tabu)
    return numerator/denominator

def grad_ttf(x0, *args):
    '''
    Gradient of Tabu Tunneling function
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
        denominator = denominator*dist
        grad_denom = grad_denom + diff/dist**2
    
    return (grad_numerator+numerator*grad_denom)/denominator


def fun_grad_ttf(x0, *args):
    '''
    pair of function value and its gradient of
    Tabu Tunneling function
    '''
    '''
    Tabu Tunneling function
    '''
    func, aspiration, tabulist = args[:3]
    fun_args = ()
    if len(args)>4:
        fun_args = tuple(args[4:])
    
    f_val, grad_val = func(x0, *fun_args)
    f_val = f_val-aspiration
    numerator = f_val**2
    grad_numerator = 2*f_val*grad_val
    denominator = 1
    grad_denominator = np.zeros_like(x0)
    for tabu in tabulist:
        diff = tabu-x0
        dist = np.linalg.norm(diff)
        denominator = denominator*dist
        grad_denominator = grad_denominator + diff/dist**2
    
    y_ttf = numerator/denominator
    deriv_y_ttf = grad_numerator/denominator+y_ttf*grad_denominator
    return y_ttf, deriv_y_ttf