import os
import sys
import unittest
import numpy as np
from scipy.optimize import approx_fprime

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack import residual_thy, residual_voigt
from TRXASprefitpack import res_grad_voigt, res_grad_thy
from TRXASprefitpack import voigt, voigt_thy
from TRXASprefitpack import edge_gaussian, edge_lorenzian

rel = 1e-4
epsilon = 1e-7

class TestGradResStatic(unittest.TestCase):

    def test_res_grad_voigt_1(self):
        '''
        Gaussian Edge
        '''
        e0_1 = 8987; e0_2 = 9000
        e0_edge = 8992
        fwhm_G_1 = 0.8
        fwhm_G_2 = 0.9
        fwhm_L_1 = 3
        fwhm_L_2 = 9
        fwhm_edge = 7
        
        # set scan range
        e = np.linspace(8960, 9020, 160)
        
        # generate model spectrum
        model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
            0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
                0.2*edge_gaussian(e-e0_edge, fwhm_edge)
        eps_static = np.ones_like(model_static)

        x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

        ref_res = np.sum(residual_voigt(x0, 2, 'g', 1, None,
        e=e, intensity=model_static, eps=eps_static)**2)/2

        ref_grad = approx_fprime(x0, lambda x0: 
        np.sum(residual_voigt(x0, 2, 'g', 1, None,
        e=e, intensity=model_static, eps=eps_static)**2)/2, 1e-7)

        res_tst, grad_tst = res_grad_voigt(x0, 2, 'g', 1, 
        None, np.zeros(len(x0), dtype=bool),
        e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_voigt_2(self):
        '''
        Lorenzian Edge
        '''
        e0_1 = 8987; e0_2 = 9000
        e0_edge = 8992
        fwhm_G_1 = 0.8
        fwhm_G_2 = 0.9
        fwhm_L_1 = 3
        fwhm_L_2 = 9
        fwhm_edge = 7
        
        # set scan range
        e = np.linspace(8960, 9020, 160)
        
        # generate model spectrum
        model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
            0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
                0.2*edge_lorenzian(e-e0_edge, fwhm_edge)
        eps_static = np.ones_like(model_static)

        x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

        ref_res = np.sum(residual_voigt(x0, 2, 'l', 1, None,
        e=e, intensity=model_static, eps=eps_static)**2)/2

        ref_grad = approx_fprime(x0, lambda x0: 
        np.sum(residual_voigt(x0, 2, 'l', 1, None,
        e=e, intensity=model_static, eps=eps_static)**2)/2, 1e-7)

        res_tst, grad_tst = res_grad_voigt(x0, 2, 'l', 1, 
        None, np.zeros(len(x0), dtype=bool),
        e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_voigt_3(self):
        '''
        Gaussian Edge + 2nd order polynomial baseline
        '''
        e0_1 = 8987; e0_2 = 9000
        e0_edge = 8992
        fwhm_G_1 = 0.8
        fwhm_G_2 = 0.9
        fwhm_L_1 = 3
        fwhm_L_2 = 9
        fwhm_edge = 7
        
        # set scan range
        e = np.linspace(8960, 9020, 160)
        
        # generate model spectrum
        model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
            0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
                0.2*edge_gaussian(e-e0_edge, fwhm_edge)+\
                3e-5*(e-e0_edge)**2+1e-3*(e-e0_edge)+1e-1
        eps_static = np.ones_like(model_static)

        x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

        ref_res = np.sum(residual_voigt(x0, 2, edge='g', num_edge=1, base_order=2,
        e=e, intensity=model_static, eps=eps_static)**2)/2

        ref_grad = approx_fprime(x0, lambda x0: 
        np.sum(residual_voigt(x0, 2, 'g', 1, 2,
        e=e, intensity=model_static, eps=eps_static)**2)/2, 1e-6)

        res_tst, grad_tst = res_grad_voigt(x0, 2, 'g', 1, 
        2, np.zeros(len(x0), dtype=bool),
        e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4, atol=1e-6)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_voigt_4(self):
        '''
        Lorenzian Edge + 2nd order polynomial baseline
        '''
        e0_1 = 8987; e0_2 = 9000
        e0_edge = 8992
        fwhm_G_1 = 0.8
        fwhm_G_2 = 0.9
        fwhm_L_1 = 3
        fwhm_L_2 = 9
        fwhm_edge = 7
        
        # set scan range
        e = np.linspace(8960, 9020, 160)
        
        # generate model spectrum
        model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
            0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
                0.2*edge_lorenzian(e-e0_edge, fwhm_edge)+\
                3e-5*(e-e0_edge)**2+1e-3*(e-e0_edge)+1e-1
        eps_static = np.ones_like(model_static)

        x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

        ref_res = np.sum(residual_voigt(x0, 2, edge='l', num_edge=1, base_order=2,
        e=e, intensity=model_static, eps=eps_static)**2)/2

        ref_grad = approx_fprime(x0, lambda x0: 
        np.sum(residual_voigt(x0, 2, 'l', 1, 2,
        e=e, intensity=model_static, eps=eps_static)**2)/2, 1e-6)

        res_tst, grad_tst = res_grad_voigt(x0, 2, 'l', 1, 
        2, np.zeros(len(x0), dtype=bool),
        e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4, atol=1e-6)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_thy_1(self):
        '''
        two thy component and two gaussian edge and 2nd order polynomial baseline (policy shift)
        '''

        e0_edge = np.array([860.5, 862])
        fwhm_edge = np.array([1, 1.5])
        peak_shift = np.array([862.5, 863])
        mixing = np.array([0.3, 0.7])
        mixing_edge = np.array([0.3, 0.7])
        fwhm_G_thy = 0.3
        fwhm_L_thy = 0.5
        
        thy_peak = np.empty(2, dtype=object)
        thy_peak[0] = np.genfromtxt(path+'/'+'Ni_example_1.stk')
        thy_peak[1] = np.genfromtxt(path+'/'+'Ni_example_2.stk')
        
        # set scan range
        e = np.linspace(852.5, 865, 51)
        
        # generate model spectrum
        model_static = mixing[0]*voigt_thy(e, thy_peak[0], fwhm_G_thy, fwhm_L_thy,
        peak_shift[0], policy='shift')+\
            mixing[1]*voigt_thy(e, thy_peak[1], fwhm_G_thy, fwhm_L_thy,
            peak_shift[1], policy='shift')+\
                mixing_edge[0]*edge_gaussian(e-e0_edge[0], fwhm_edge[0])+\
                    mixing_edge[1]*edge_gaussian(e-e0_edge[1], fwhm_edge[1])+\
                    1e-2*(e-860)**2 + 2e-1*(e-860) + 3e-1
        
        eps_static = np.ones_like(model_static)

        x0 = [0.2, 0.4, 862.2, 863.3, 860, 861.5, 0.8, 1.2]

        ref_res = np.sum(residual_thy(x0, 'shift', thy_peak,
        'g', 2, 2, e=e, intensity=model_static, eps=eps_static)**2)/2
        ref_grad = approx_fprime(x0, lambda x0: 1/2*np.sum(residual_thy(x0, 'shift', thy_peak,
        'g', 2, 2, e=e, intensity=model_static, eps=eps_static)**2), 1e-6)

        res_tst, grad_tst = res_grad_thy(x0, 'shift', thy_peak, 'g', 2, 2,
        np.zeros(len(x0), dtype=bool), e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4, atol=1e-6)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_thy_2(self):
        '''
        two thy component and two gaussian edge and 2nd order polynomial baseline (policy scale)
        '''

        e0_edge = np.array([860.5, 862])
        fwhm_edge = np.array([1, 1.5])
        peak_shift = np.array([862.5, 863])
        mixing = np.array([0.3, 0.7])
        mixing_edge = np.array([0.3, 0.7])
        fwhm_G_thy = 0.3
        fwhm_L_thy = 0.5
        
        thy_peak = np.empty(2, dtype=object)
        thy_peak[0] = np.genfromtxt(path+'/'+'Ni_example_1.stk')
        thy_peak[1] = np.genfromtxt(path+'/'+'Ni_example_2.stk')
        
        # set scan range
        e = np.linspace(852.5, 865, 51)
        
        # generate model spectrum
        model_static = mixing[0]*voigt_thy(e, thy_peak[0], fwhm_G_thy, fwhm_L_thy,
        peak_shift[0], policy='shift')+\
            mixing[1]*voigt_thy(e, thy_peak[1], fwhm_G_thy, fwhm_L_thy,
            peak_shift[1], policy='shift')+\
                mixing_edge[0]*edge_gaussian(e-e0_edge[0], fwhm_edge[0])+\
                    mixing_edge[1]*edge_gaussian(e-e0_edge[1], fwhm_edge[1])+\
                    1e-2*(e-860)**2 + 2e-1*(e-860) + 3e-1
        
        eps_static = np.ones_like(model_static)

        x0 = [0.2, 0.4, 1.01, 1.05, 1.01, 860, 862.5, 0.8, 1.2]

        ref_res = np.sum(residual_thy(x0, 'scale', thy_peak,
        'g', 2, 2, e=e, intensity=model_static, eps=eps_static)**2)/2
        ref_grad = approx_fprime(x0, lambda x0: 1/2*np.sum(residual_thy(x0, 'scale', thy_peak,
        'g', 2, 2, e=e, intensity=model_static, eps=eps_static)**2), 1e-6)

        res_tst, grad_tst = res_grad_thy(x0, 'scale', thy_peak, 'g', 2, 2,
        np.zeros(len(x0), dtype=bool), e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4, atol=1e-6)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_thy_3(self):
        '''
        two thy component and two gaussian edge and 2nd order polynomial baseline (policy both)
        '''

        e0_edge = np.array([860.5, 862])
        fwhm_edge = np.array([1, 1.5])
        peak_shift = np.array([862.5, 863])
        mixing = np.array([0.3, 0.7])
        mixing_edge = np.array([0.3, 0.7])
        fwhm_G_thy = 0.3
        fwhm_L_thy = 0.5
        
        thy_peak = np.empty(2, dtype=object)
        thy_peak[0] = np.genfromtxt(path+'/'+'Ni_example_1.stk')
        thy_peak[1] = np.genfromtxt(path+'/'+'Ni_example_2.stk')
        
        # set scan range
        e = np.linspace(852.5, 865, 51)
        
        # generate model spectrum
        model_static = mixing[0]*voigt_thy(e, thy_peak[0], fwhm_G_thy, fwhm_L_thy,
        peak_shift[0], policy='shift')+\
            mixing[1]*voigt_thy(e, thy_peak[1], fwhm_G_thy, fwhm_L_thy,
            peak_shift[1], policy='shift')+\
                mixing_edge[0]*edge_gaussian(e-e0_edge[0], fwhm_edge[0])+\
                    mixing_edge[1]*edge_gaussian(e-e0_edge[1], fwhm_edge[1])+\
                    1e-2*(e-860)**2 + 2e-1*(e-860) + 3e-1
        
        eps_static = np.ones_like(model_static)

        x0 = [0.2, 0.4, 862.2, 863.3, 1.01, 1.05, 860.5, 862.5, 0.8, 1.2]

        ref_res = np.sum(residual_thy(x0, 'both', thy_peak,
        'g', 2, 2, e=e, intensity=model_static, eps=eps_static)**2)/2
        ref_grad = approx_fprime(x0, lambda x0: 1/2*np.sum(residual_thy(x0, 'both', thy_peak,
        'g', 2, 2, e=e, intensity=model_static, eps=eps_static)**2), 1e-6)

        res_tst, grad_tst = res_grad_thy(x0, 'both', thy_peak, 'g', 2, 2,
        np.zeros(len(x0), dtype=bool), e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4, atol=1e-6)

        self.assertEqual((cond_res, cond_grad), (True, True))

    def test_res_grad_thy_4(self):
        '''
        two thy component and two lorenzian edge and 2nd order polynomial baseline (policy both)
        '''

        e0_edge = np.array([860.5, 862])
        fwhm_edge = np.array([1, 1.5])
        peak_shift = np.array([862.5, 863])
        mixing = np.array([0.3, 0.7])
        mixing_edge = np.array([0.3, 0.7])
        fwhm_G_thy = 0.3
        fwhm_L_thy = 0.5
        
        thy_peak = np.empty(2, dtype=object)
        thy_peak[0] = np.genfromtxt(path+'/'+'Ni_example_1.stk')
        thy_peak[1] = np.genfromtxt(path+'/'+'Ni_example_2.stk')
        
        # set scan range
        e = np.linspace(852.5, 865, 51)
        
        # generate model spectrum
        model_static = mixing[0]*voigt_thy(e, thy_peak[0], fwhm_G_thy, fwhm_L_thy,
        peak_shift[0], policy='shift')+\
            mixing[1]*voigt_thy(e, thy_peak[1], fwhm_G_thy, fwhm_L_thy,
            peak_shift[1], policy='shift')+\
                mixing_edge[0]*edge_lorenzian(e-e0_edge[0], fwhm_edge[0])+\
                    mixing_edge[1]*edge_lorenzian(e-e0_edge[1], fwhm_edge[1])+\
                    1e-2*(e-860)**2 + 2e-1*(e-860) + 3e-1
        
        eps_static = np.ones_like(model_static)

        x0 = [0.2, 0.4, 862.2, 863.3, 1.01, 1.05, 860.5, 862.5, 0.8, 1.2]

        ref_res = np.sum(residual_thy(x0, 'both', thy_peak,
        'l', 2, 2, e=e, intensity=model_static, eps=eps_static)**2)/2
        ref_grad = approx_fprime(x0, lambda x0: 1/2*np.sum(residual_thy(x0, 'both', thy_peak,
        'l', 2, 2, e=e, intensity=model_static, eps=eps_static)**2), 1e-6)

        res_tst, grad_tst = res_grad_thy(x0, 'both', thy_peak, 'l', 2, 2,
        np.zeros(len(x0), dtype=bool), e=e, intensity=model_static, eps=eps_static)

        cond_res = np.allclose(res_tst, ref_res)
        cond_grad = np.allclose(grad_tst, ref_grad, rtol=1e-4, atol=1e-6)

        self.assertEqual((cond_res, cond_grad), (True, True))

if __name__ == '__main__':
    unittest.main()      