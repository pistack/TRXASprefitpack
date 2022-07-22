'''
res:
subpackage for resdiual function for
fitting time delay scan data or static spectrum data

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from .parm_bound import set_bound_t0, set_bound_tau
from .res_gen import res_scan, res_lmfit, residual_scalar
from .res_gen import jac_scan, jac_lmfit, grad_res_scalar
from .res_decay import residual_decay
from .res_decay import jac_res_decay
from .res_osc import residual_dmp_osc
from .res_osc import jac_res_dmp_osc
from .res_both import residual_both
from .res_both import jac_res_both
from .res_voigt import residual_voigt
from .res_voigt import jac_res_voigt
from .res_thy import residual_thy
from .res_thy import jac_res_thy

__all__ = ['set_bound_t0', 'set_bound_tau',
           'res_scan', 'jac_scan',
           'res_lmfit', 'jac_lmfit',
           'residual_scalar', 'grad_res_scalar',
           'residual_decay', 'jac_res_decay',
           'residual_dmp_osc', 'jac_res_dmp_osc',
           'residual_both', 'jac_res_both',
           'residual_voigt', 'jac_res_voigt',
           'residual_thy', 'jac_res_thy']