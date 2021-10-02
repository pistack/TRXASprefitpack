'''
data_process:

subpackage for data processing

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

from .scale import automate_scaling
from .corr import corr_a_method

__all__ = ['automate_scaling', 'corr_a_method']
