import os
import sys
import unittest
import numpy as np
from scipy.optimize import approx_fprime

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack import residual_decay, residual_dmp_osc, residual_both
from TRXASprefitpack import res_grad_decay, res_grad_dmp_osc, res_grad_both
from TRXASprefitpack import solve_seq_model, rate_eq_conv
from TRXASprefitpack import dmp_osc_conv

rel = 1e-4
epsilon = 1e-7

class TestGradResTransient(unittest.TestCase):

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
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
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
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_decay(x0, 3, False, 'g', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon) # noise field

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
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
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
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_decay(x0, 3, False, 'c', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon) # noise field

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
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
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
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_decay(x0, 3, True, 'pv', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon) # noise field

        self.assertEqual((check_res, check_grad), (True, True))
    
    def test_res_grad_dmp_osc_1(self):
        '''
        Gaussian IRF
        '''
        fwhm = 0.100
        tau = np.array([0.5, 10, 1000])
        period = np.array([0.2, 3, 200])
        phase = np.array([np.pi/17, -np.pi/6, np.pi/2])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = np.array([1, 1, 1])
        abs_2 = np.array([0.5, 0.8, 0.2])
        abs_3 = np.array([-0.5, 0.7, 0.9])
        abs_4 = np.array([0.6, 0.3, -1])
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = dmp_osc_conv(t_seq-t0[0], fwhm, tau, period, phase, abs_1, irf='g')
        y_obs_2 = dmp_osc_conv(t_seq-t0[1], fwhm, tau, period, phase, abs_2, irf='g')
        y_obs_3 = dmp_osc_conv(t_seq-t0[2], fwhm, tau, period, phase, abs_3, irf='g')
        y_obs_4 = dmp_osc_conv(t_seq-t0[3], fwhm, tau, period, phase, abs_4, irf='g')
        
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

        x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990, 0.3, 5, 250])

        res_ref = 1/2*np.sum(residual_dmp_osc(x0, 3, 'g', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_dmp_osc(x0, 3, 'g', 
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_dmp_osc(x0, 3, 'g', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_dmp_osc_2(self):
        '''
        Cauchy IRF
        '''
        fwhm = 0.100
        tau = np.array([0.5, 10, 1000])
        period = np.array([0.2, 3, 200])
        phase = np.array([np.pi/17, -np.pi/6, np.pi/2])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = np.array([1, 1, 1])
        abs_2 = np.array([0.5, 0.8, 0.2])
        abs_3 = np.array([-0.5, 0.7, 0.9])
        abs_4 = np.array([0.6, 0.3, -1])
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = dmp_osc_conv(t_seq-t0[0], fwhm, tau, period, phase, abs_1, irf='c')
        y_obs_2 = dmp_osc_conv(t_seq-t0[1], fwhm, tau, period, phase, abs_2, irf='c')
        y_obs_3 = dmp_osc_conv(t_seq-t0[2], fwhm, tau, period, phase, abs_3, irf='c')
        y_obs_4 = dmp_osc_conv(t_seq-t0[3], fwhm, tau, period, phase, abs_4, irf='c')
        
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

        x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990, 0.3, 5, 250])

        res_ref = 1/2*np.sum(residual_dmp_osc(x0, 3, 'c', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_dmp_osc(x0, 3, 'c', 
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_dmp_osc(x0, 3, 'c', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon) # noise field

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_dmp_osc_3(self):
        '''
        Pseudo Voigt IRF
        '''
        fwhm = 0.100; eta = 0.7
        tau = np.array([0.5, 10, 1000])
        period = np.array([0.2, 3, 200])
        phase = np.array([np.pi/17, -np.pi/6, np.pi/2])
        
        # set time range (mixed step)
        t_seq1 = np.arange(-2, -1, 0.2)
        t_seq2 = np.arange(-1, 2, 0.02)
        t_seq3 = np.arange(2, 5, 0.2)
        t_seq4 = np.arange(5, 10, 1)
        t_seq5 = np.arange(10, 100, 10)
        t_seq6 = np.arange(100, 1000, 100)
        t_seq7 = np.linspace(1000, 2000, 2)
        
        t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
        
        # Now generates measured transient signal
        # Last element is ground state
        abs_1 = np.array([1, 1, 1])
        abs_2 = np.array([0.5, 0.8, 0.2])
        abs_3 = np.array([-0.5, 0.7, 0.9])
        abs_4 = np.array([0.6, 0.3, -1])
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = dmp_osc_conv(t_seq-t0[0], fwhm, tau, period, phase, abs_1, 
        irf='pv', eta=eta)
        y_obs_2 = dmp_osc_conv(t_seq-t0[1], fwhm, tau, period, phase, abs_2, 
        irf='pv', eta=eta)
        y_obs_3 = dmp_osc_conv(t_seq-t0[2], fwhm, tau, period, phase, abs_3, 
        irf='pv', eta=eta)
        y_obs_4 = dmp_osc_conv(t_seq-t0[3], fwhm, tau, period, phase, abs_4, 
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

        x0 = np.array([0.15, 0.23, 0, 0, 0, 0, 0.4, 9, 990,  0.3, 5, 250])

        res_ref = 1/2*np.sum(residual_dmp_osc(x0, 3, 'pv', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_dmp_osc(x0, 3, 'pv', 
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_dmp_osc(x0, 3, 'pv', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon) # noise field

        self.assertEqual((check_res, check_grad), (True, True))


    def test_res_grad_both_1(self):
        '''
        Gaussian IRF
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100
        tau_osc = np.array([0.3, 8, 800])
        period_osc = np.array([0.2, 2, 300])
        phase = np.array([np.pi/17, -np.pi/6, np.pi/2])
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
        abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.5*np.array([0.6, 0.1, 0.3])
        abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.1*np.array([0.2, 0.17, 1.19])
        abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.2*np.array([0.16, 0.28, 0.339])
        abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.18*np.array([1, 3, -2])
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_1_osc, irf='g')
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_2_osc, irf='g')
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_3_osc, irf='g')
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_4_osc, irf='g')
        
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

        x0 = np.array([0.15, 0, 0, 0, 0, 0.5, 9, 990, 1, 15, 1500, 0.5, 1, 200])

        res_ref = 1/2*np.sum(residual_both(x0, 3, 3, False, 'g', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_both(x0, 3, 3, False, 'g', 
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_both(x0, 3, 3, False, 'g', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon)

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_both_2(self):
        '''
        Cauchy IRF
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100
        tau_osc = np.array([0.3, 8, 800])
        period_osc = np.array([0.2, 2, 300])
        phase = np.array([np.pi/17, -np.pi/6, np.pi/2])
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
        abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.5*np.array([0.6, 0.1, 0.3])
        abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.1*np.array([0.2, 0.17, 1.19])
        abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.2*np.array([0.16, 0.28, 0.339])
        abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.18*np.array([1, 3, -2])
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='c')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_1_osc, irf='c')
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='c')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_2_osc, irf='c')
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='c')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_3_osc, irf='c')
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='c')+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_4_osc, irf='c')
        
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

        x0 = np.array([0.15, 0, 0, 0, 0, 0.5, 9, 990, 1, 15, 1500, 0.5, 1, 200])

        res_ref = 1/2*np.sum(residual_both(x0, 3, 3, False, 'c', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_both(x0, 3, 3, False, 'c', 
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_both(x0, 3, 3, False, 'c', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon)

        self.assertEqual((check_res, check_grad), (True, True))

    def test_res_grad_both_3(self):
        '''
        Pseudo Voigt IRF
        '''
        tau_1 = 0.5; tau_2 = 10
        tau_3 = 1000; fwhm = 0.100; eta = 0.7
        tau_osc = np.array([0.3, 8, 800])
        period_osc = np.array([0.2, 2, 300])
        phase = np.array([np.pi/17, -np.pi/6, np.pi/2])
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
        abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.5*np.array([0.6, 0.1, 0.3])
        abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.1*np.array([0.2, 0.17, 1.19])
        abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.2*np.array([0.16, 0.28, 0.339])
        abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.18*np.array([1, 3, -2])
        
        t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan
        
        # generate measured data
        y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_1_osc, 
            irf='pv', eta=eta)
        y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_2_osc, 
            irf='pv', eta=eta)
        y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_3_osc, 
            irf='pv', eta=eta)
        y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, 
        irf='pv', eta=eta)+\
            dmp_osc_conv(t_seq-t0[1], fwhm, tau_osc, period_osc, phase, abs_4_osc, 
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

        x0 = np.array([0.15, 0.3 , 0, 0, 0, 0, 0.5, 9, 990, 1, 15, 1500, 0.5, 1, 200])

        res_ref = 1/2*np.sum(residual_both(x0, 3, 3, False, 'pv', 
        t=t, intensity=intensity, eps=eps)**2)
        grad_ref = approx_fprime(x0, lambda x0: \
            1/2*np.sum(residual_both(x0, 3, 3, False, 'pv', 
            t=t, intensity=intensity, eps=eps)**2), epsilon)

        res_tst, grad_tst = res_grad_both(x0, 3, 3, False, 'pv', 
        np.zeros_like(x0, dtype=bool), t, intensity, eps)

        check_res = np.allclose(res_ref, res_tst)
        check_grad = np.allclose(grad_ref, grad_tst, rtol=rel,
        atol=epsilon)

        self.assertEqual((check_res, check_grad), (True, True))


if __name__ == '__main__':
    unittest.main()