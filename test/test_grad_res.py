import os
import sys
import unittest
import numpy as np
from scipy.optimize import approx_fprime

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack import residual_decay, residual_dmp_osc, residual_both
from TRXASprefitpack import residual_voigt, residual_thy
from TRXASprefitpack import res_grad_decay, res_grad_dmp_osc, res_grad_both
from TRXASprefitpack import res_grad_voigt, res_grad_thy
from TRXASprefitpack import solve_seq_model, rate_eq_conv
from TRXASprefitpack import dmp_osc_conv_gau, dmp_osc_conv_cauchy, dmp_osc_conv_pvoigt 

class TestGradRes(unittest.TestCase):

    def test_res_grad_decay_1(self):
        '''
        Test res_grad_decay (irf: g)
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100
        # initial condition
        y0 = np.array([1, 0, 0, 0])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = [1, 1, 1, 0]
        abs_2 = [0.5, 0.8, 0.2, 0]
        abs_3 = [-0.5, 0.7, 0.9, 0]
        abs_4 = [0.6, 0.3, -1, 0]
        
        t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')
        
        eps_obs_1 = np.ones_like(y_obs_1)
        eps_obs_2 = np.ones_like(y_obs_2)
        eps_obs_3 = np.ones_like(y_obs_3)
        eps_obs_4 = np.ones_like(y_obs_4)
        
        # generate measured intensity
        i_obs_1 = y_obs_1
        i_obs_2 = y_obs_2
        i_obs_3 = y_obs_3
        i_obs_4 = y_obs_4

        t = [t_seq] 
        intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
        eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

        x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990])

        res_ref = 1/2*np.sum(residual_decay(x0, False, 'g', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_decay(x0, False, 'g', 
            t=t, intensity=intensity, eps=eps)**2), 1e-6)

        res_tst, grad_tst = res_grad_decay(x0, 3, False, 'g', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=1e-3,
        atol=1e-6) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_decay_2(self):
        '''
        Test res_grad_decay (irf: c)
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100
        # initial condition
        y0 = np.array([1, 0, 0, 0])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = [1, 1, 1, 0]
        abs_2 = [0.5, 0.8, 0.2, 0]
        abs_3 = [-0.5, 0.7, 0.9, 0]
        abs_4 = [0.6, 0.3, -1, 0]
        
        t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='c')
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='c')
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='c')
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='c')
        
        eps_obs_1 = np.ones_like(y_obs_1)
        eps_obs_2 = np.ones_like(y_obs_2)
        eps_obs_3 = np.ones_like(y_obs_3)
        eps_obs_4 = np.ones_like(y_obs_4)
        
        # generate measured intensity
        i_obs_1 = y_obs_1
        i_obs_2 = y_obs_2
        i_obs_3 = y_obs_3 
        i_obs_4 = y_obs_4 

        t = [t_seq] 
        intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
        eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

        x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990])

        res_ref = 1/2*np.sum(residual_decay(x0, False, 'c', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_decay(x0, False, 'c', 
            t=t, intensity=intensity, eps=eps)**2), 1e-6)

        res_tst, grad_tst = res_grad_decay(x0, 3, False, 'c', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=1e-3,
        atol=1e-6) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_decay_3(self):
        '''
        Test res_grad_decay (irf: pv)
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100; eta = 0.7
        # initial condition
        y0 = np.array([1, 0, 0, 0])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = [1, 1, 1, 0]
        abs_2 = [0.5, 0.8, 0.2, 0]
        abs_3 = [-0.5, 0.7, 0.9, 0]
        abs_4 = [0.6, 0.3, -1, 0]
        
        t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)
        
        eps_obs_1 = np.ones_like(y_obs_1)
        eps_obs_2 = np.ones_like(y_obs_2)
        eps_obs_3 = np.ones_like(y_obs_3)
        eps_obs_4 = np.ones_like(y_obs_4)
        
        # generate measured intensity
        i_obs_1 = y_obs_1
        i_obs_2 = y_obs_2 
        i_obs_3 = y_obs_3 
        i_obs_4 = y_obs_4 

        t = [t_seq] 
        intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
        eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

        x0 = np.array([0.3, 0.15, 0, 0, 0, 0, 0.4, 9, 990])

        res_ref = 1/2*np.sum(residual_decay(x0, True, 'pv', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_decay(x0, True, 'pv', 
            t=t, intensity=intensity, eps=eps)**2), 1e-6)

        res_tst, grad_tst = res_grad_decay(x0, 3, True, 'pv', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=1e-3,
        atol=1e-6) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_both_1(self):
        '''
        Gaussian IRF
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100
        tau_osc = 1; period_osc = 0.3
        phase = np.pi/4
        # initial condition
        y0 = np.array([1, 0, 0, 0])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.05
        abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.001
        abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.002
        abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.0018
        
        t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')+\
            abs_1_osc*dmp_osc_conv_gau(t_seq-t0[0], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')+\
            abs_2_osc*dmp_osc_conv_gau(t_seq-t0[1], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')+\
            abs_3_osc*dmp_osc_conv_gau(t_seq-t0[2], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')+\
            abs_4_osc*dmp_osc_conv_gau(t_seq-t0[3], fwhm, 1/tau_osc, period_osc, phase)
        
        eps_obs_1 = np.ones_like(y_obs_1)
        eps_obs_2 = np.ones_like(y_obs_2)
        eps_obs_3 = np.ones_like(y_obs_3)
        eps_obs_4 = np.ones_like(y_obs_4)
        
        # generate measured intensity
        i_obs_1 = y_obs_1 
        i_obs_2 = y_obs_2 
        i_obs_3 = y_obs_3 
        i_obs_4 = y_obs_4 

        t = [t_seq] 
        intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
        eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

        x0 = np.array([0.15, 0, 0, 0, 0, 0.5, 9, 990, 1.5, 0.5])

        res_ref = 1/2*np.sum(residual_both(x0, 3, 1, False, 'g', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_both(x0, 3, 1, False, 'g', 
            t=t, intensity=intensity, eps=eps)**2), 1e-6)

        res_tst, grad_tst = res_grad_both(x0, 3, 1, False, 'g', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=1e-3,
        atol=1e-6) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_both_2(self):
        '''
        Cauchy IRF
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100
        tau_osc = 1; period_osc = 0.3
        phase = np.pi/4
        # initial condition
        y0 = np.array([1, 0, 0, 0])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.05
        abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.001
        abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.002
        abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.0018
        
        t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_1_osc*dmp_osc_conv_gau(t_seq-t0[0], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_2_osc*dmp_osc_conv_gau(t_seq-t0[1], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_3_osc*dmp_osc_conv_gau(t_seq-t0[2], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_4_osc*dmp_osc_conv_gau(t_seq-t0[3], fwhm, 1/tau_osc, period_osc, phase)
        
        eps_obs_1 = np.ones_like(y_obs_1)
        eps_obs_2 = np.ones_like(y_obs_2)
        eps_obs_3 = np.ones_like(y_obs_3)
        eps_obs_4 = np.ones_like(y_obs_4)
        
        # generate measured intensity
        i_obs_1 = y_obs_1 
        i_obs_2 = y_obs_2 
        i_obs_3 = y_obs_3 
        i_obs_4 = y_obs_4 

        t = [t_seq] 
        intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
        eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

        x0 = np.array([0.15, 0, 0, 0, 0, 0.5, 9, 990, 1.5, 0.5])

        res_ref = 1/2*np.sum(residual_both(x0, 3, 1, False, 'c', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_both(x0, 3, 1, False, 'c', 
            t=t, intensity=intensity, eps=eps)**2), 1e-6)

        res_tst, grad_tst = res_grad_both(x0, 3, 1, False, 'c', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=1e-3,
        atol=1e-6) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_both_3(self):
        '''
        Pseudo Voigt IRF
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100; eta = 0.7
        tau_osc = 1; period_osc = 0.3
        phase = np.pi/4
        # initial condition
        y0 = np.array([1, 0, 0, 0])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.05
        abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.001
        abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.002
        abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.0018
        
        t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)+\
            abs_1_osc*dmp_osc_conv_gau(t_seq-t0[0], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_2_osc*dmp_osc_conv_gau(t_seq-t0[1], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_3_osc*dmp_osc_conv_gau(t_seq-t0[2], fwhm, 1/tau_osc, period_osc, phase)
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='c')+\
            abs_4_osc*dmp_osc_conv_gau(t_seq-t0[3], fwhm, 1/tau_osc, period_osc, phase)
        
        eps_obs_1 = np.ones_like(y_obs_1)
        eps_obs_2 = np.ones_like(y_obs_2)
        eps_obs_3 = np.ones_like(y_obs_3)
        eps_obs_4 = np.ones_like(y_obs_4)
        
        # generate measured intensity
        i_obs_1 = y_obs_1 
        i_obs_2 = y_obs_2 
        i_obs_3 = y_obs_3 
        i_obs_4 = y_obs_4 

        t = [t_seq] 
        intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
        eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

        x0 = np.array([0.15, 0, 0, 0, 0, 0.5, 9, 990, 1.5, 0.5])

        res_ref = 1/2*np.sum(residual_both(x0, 3, 1, False, 'c', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_both(x0, 3, 1, False, 'c', 
            t=t, intensity=intensity, eps=eps)**2), 1e-6)

        res_tst, grad_tst = res_grad_both(x0, 3, 1, False, 'c', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=1e-3,
        atol=1e-6) # noise field

        self.assertEqual((check_res, check_grad), (True, True))


if __name__ == '__main__':
    unittest.main()