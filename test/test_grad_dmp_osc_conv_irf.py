import os
import sys
import unittest
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack import test_num_deriv
from TRXASprefitpack import dmp_osc_conv_gau, dmp_osc_conv_cauchy
from TRXASprefitpack import deriv_dmp_osc_conv_gau, deriv_dmp_osc_conv_cauchy

class TestDerivExpConvIRF(unittest.TestCase):
    def test_deriv_dmp_osc_conv_gau_1(self):
        tau = 1
        fwhm = 0.15
        period = 1
        phase = 0
        t = np.linspace(-1, 100, 2001)
        tst = deriv_dmp_osc_conv_gau(t, fwhm, 1/tau, period, phase)
        ref = test_num_deriv(dmp_osc_conv_gau, t, fwhm, 1/tau, period, phase, eps=5e-8)
        result = np.isclose(tst, ref)
        result = np.allclose(tst, ref)
        self.assertEqual(result, True)

    def test_deriv_dmp_osc_conv_cauchy_1(self):
        tau = 1
        fwhm = 0.5
        period = 1
        phase = np.pi/4
        t = np.linspace(-1, 100, 2001)
        tst = deriv_dmp_osc_conv_cauchy(t, fwhm, 1/tau, period, phase)
        ref = test_num_deriv(dmp_osc_conv_cauchy, t, fwhm, 1/tau, period, phase, eps=5e-8)
        result = np.allclose(tst, ref)
        self.assertEqual(result, True)

if __name__ == '__main__':
    unittest.main()