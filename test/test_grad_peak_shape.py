import os
import sys
import unittest
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.mathfun.deriv_tst import test_num_deriv
from TRXASprefitpack import edge_gaussian, edge_lorenzian
from TRXASprefitpack import deriv_edge_gaussian, deriv_edge_lorenzian
from TRXASprefitpack import voigt, voigt_thy
from TRXASprefitpack import deriv_voigt, deriv_voigt_thy

class TestDerivPeakShape(unittest.TestCase):

    def test_deriv_voigt_1(self):
        '''
        Test gradient of voigt function (when fwhm_G == 0)
        '''
        e = np.linspace(-5, 5, 1001)
        fwhm_L = 1
        tst = deriv_voigt(e, 0, fwhm_L)
        ref = test_num_deriv(voigt, e, 0, fwhm_L)
        result = np.allclose(tst[:, [0, 2]], ref[:, [0, 2]])
        self.assertEqual(result, True)

    def test_deriv_voigt_2(self):
        '''
        Test gradient of voigt function (when fwhm_L == 0)
        '''
        e = np.linspace(-5, 5, 1001)
        fwhm_G = 1
        tst = deriv_voigt(e, fwhm_G, 0)
        ref = test_num_deriv(voigt, e, fwhm_G, 0)
        result = np.allclose(tst[:, [0, 1]], ref[:, [0, 1]])
        self.assertEqual(result, True)

    def test_deriv_voigt_3(self):
        '''
        Test gradient of voigt function (when fwhm_L == 0)
        '''
        e = np.linspace(-5, 5, 1001)
        fwhm_G = 1; fwhm_L = 0.5
        tst = deriv_voigt(e, fwhm_G, fwhm_L)
        ref = test_num_deriv(voigt, e, fwhm_G, fwhm_L)
        result = np.allclose(tst, ref)
        self.assertEqual(result, True)
    
    def test_deriv_edge_1(self):
        '''
        Test gradient of gaussian type edge
        '''
        e = np.linspace(-5, 5, 1001)
        fwhm_G = 1
        tst = deriv_edge_gaussian(e, fwhm_G)
        ref = test_num_deriv(edge_gaussian,
        e, fwhm_G)
        result = np.allclose(tst, ref)
        self.assertEqual(result, True)

    def test_deriv_edge_2(self):
        '''
        Test gradient of lorenzian type edge
        '''
        e = np.linspace(-5, 5, 1001)
        fwhm_L = 1
        tst = deriv_edge_lorenzian(e, fwhm_L)
        ref = test_num_deriv(edge_lorenzian,
        e, fwhm_L)
        result = np.allclose(tst, ref)
        self.assertEqual(result, True)
    
    def test_deriv_voigt_thy_1(self):
        '''
        Test gradient of voigt_thy (policy == 'shift')
        '''
        e = np.linspace(-10, 20, 300)
        thy_peak = \
            np.array([[-3, -2, 1, 0, 3, 3.1, 3.5, 4, 10, 15], 
            [1e-2, 1e-2, 1e-1, 2e-2, 5e-2, 3e-2, 6e-2, 1e-3, 1e-2, 1.5e-2]]).T
        tst = deriv_voigt_thy(e, thy_peak, 0.3, 0.5, -1.5, 'shift')
        ref = test_num_deriv(lambda e, fwhm_G, fwhm_L, peak_shift: \
            voigt_thy(e, thy_peak, fwhm_G, fwhm_L, peak_shift, 'shift'),
            e, 0.3, 0.5, -1.5)
        result = np.allclose(tst, ref[:, 1:])
        self.assertEqual(result, True)

    def test_deriv_voigt_thy_2(self):
        '''
        Test gradient of voigt_thy (policy == 'scale')
        '''
        e = np.linspace(-10, 20, 300)
        thy_peak = \
            np.array([[-3, -2, 1, 0, 3, 3.1, 3.5, 4, 10, 15], 
            [1e-2, 1e-2, 1e-1, 2e-2, 5e-2, 3e-2, 6e-2, 1e-3, 1e-2, 1.5e-2]]).T
        tst = deriv_voigt_thy(e, thy_peak, 0.3, 0.5, 0.996, 'scale')
        ref = test_num_deriv(lambda e, fwhm_G, fwhm_L, peak_scale: \
            voigt_thy(e, thy_peak, fwhm_G, fwhm_L, peak_scale, 'scale'),
            e, 0.3, 0.5, 0.996)
        result = np.allclose(tst, ref[:, 1:])
        self.assertEqual(result, True)

    def test_deriv_voigt_thy_3(self):
        '''
        Test gradient of voigt_thy (policy == 'both')
        '''
        e = np.linspace(-10, 20, 300)
        thy_peak = \
            np.array([[-3, -2, 1, 0, 3, 3.1, 3.5, 4, 10, 15], 
            [1e-2, 1e-2, 1e-1, 2e-2, 5e-2, 3e-2, 6e-2, 1e-3, 1e-2, 1.5e-2]]).T
        tst = deriv_voigt_thy(e, thy_peak, 0.3, 0.5, 
        np.array([-1.5, 0.996]), 'both')
        ref = test_num_deriv(lambda e, fwhm_G, fwhm_L, peak_shift, peak_scale: \
            voigt_thy(e, thy_peak, fwhm_G, fwhm_L, 
            np.array([peak_shift, peak_scale]), 'both'),
            e, 0.3, 0.5, -1.5, 0.996)
        result = np.allclose(tst, ref[:, 1:])
        self.assertEqual(result, True)




if __name__ == '__main__':
    unittest.main()
