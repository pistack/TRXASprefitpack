'''
mathfun:
subpackage for the mathematical functions for TRXASprefitpack

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from .broad import gen_theory_data
from .irf import gau_irf, cauchy_irf, pvoigt_irf, calc_eta
from .exp_conv_irf import exp_conv_gau, exp_conv_cauchy, exp_conv_pvoigt
from .exp_conv_irf import dmp_osc_conv_gau, dmp_osc_conv_cauchy, dmp_osc_conv_pvoigt
from .rate_eq import solve_model, solve_seq_model, solve_l_model, compute_model
from .rate_eq import compute_signal_gau, compute_signal_cauchy
from .rate_eq import compute_signal_pvoigt
from .exp_decay_fit import model_n_comp_conv, fact_anal_exp_conv
from .exp_decay_fit import rate_eq_conv, fact_anal_rate_eq_conv
from .exp_decay_fit import dmp_osc_conv, fact_anal_dmp_osc_conv


__all__ = ['gen_theory_data',
           'gau_irf', 'cauchy_irf', 'pvoigt_irf', 'calc_eta',
           'exp_conv_gau', 'exp_conv_cauchy', 'exp_conv_pvoigt',
           'dmp_osc_conv_gau', 'dmp_osc_conv_cauchy', 'dmp_osc_conv_pvoigt',
           'solve_model', 'solve_seq_model', 'solve_l_model', 'compute_model',
           'compute_signal_gau', 'compute_signal_cauchy', 'compute_signal_pvoigt', 
           'model_n_comp_conv', 'fact_anal_exp_conv',
           'rate_eq_conv', 'fact_anal_rate_eq_conv',
           'dmp_osc_conv', 'fact_anal_dmp_osc_conv']
