'''
mathfun:
subpackage for the mathematical functions for TRXASprefitpack

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from .irf import gau_irf, cauchy_irf, pvoigt_irf
from .exp_conv_irf import exp_conv_gau, exp_conv_cauchy, exp_conv_pvoigt
from .rate_eq import solve_model, compute_model
from .rate_eq import compute_signal_gau, compute_signal_cauchy
from .rate_eq import compute_signal_pvoigt
from .exp_decay_fit import model_n_comp_conv, fact_anal_exp_conv


__all__ = ['gau_irf', 'cauchy_irf', 'pvoigt_irf',
           'exp_conv_gau', 'exp_conv_cauchy', 'exp_conv_pvoigt',
           'solve_model', 'compute_model',
           'compute_signal_gau', 'compute_signal_cauchy',
           'compute_signal_pvoigt', 'model_n_comp_conv', 'fact_anal_exp_conv']
