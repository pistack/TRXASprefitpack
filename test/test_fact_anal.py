import os
import sys
import unittest
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack import exp_conv, fact_anal_exp_conv
from TRXASprefitpack import dmp_osc_conv, fact_anal_dmp_osc_conv
from TRXASprefitpack import sum_exp_dmp_osc_conv, fact_anal_sum_exp_dmp_osc_conv

class TestFactAnal(unittest.TestCase):

    def test_fact_anal_exp_conv(self):
        fwhm = 0.15
        c_ref = np.array([1, -0.5, -0.25])
        tau = np.array([0.3, 10])
        t = np.hstack((np.arange(-1, 1, 0.05), np.linspace(1, 100, 99)))
        model = exp_conv(t, fwhm, tau, c_ref, True, irf='g')
        noise = np.random.normal(0, np.max(np.abs(model))/10, model.size)
        eps = np.max(np.abs(model))/10*np.ones_like(noise)
        expt = model+noise
        c_tst = fact_anal_exp_conv(t, fwhm, tau, True, irf='g', intensity=expt, eps=eps)
        cond = np.linalg.norm(c_ref-c_tst)/(np.linalg.norm(c_ref)) < 1e-1
        self.assertEqual(cond, True)

    def test_fact_anal_dmp_osc_conv(self):
        fwhm = 0.15
        c_ref = np.array([1, -0.5])
        tau = np.array([0.3, 1])
        period = np.array([0.5, 1])
        phase = np.array([np.pi/3, np.pi/4])
        t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
        model = dmp_osc_conv(t, fwhm, tau, period, phase, c_ref, irf='g')
        noise = np.random.normal(0, np.max(np.abs(model))/100, model.size)
        eps = np.max(np.abs(model))/100*np.ones_like(noise)
        expt = model+noise
        c_tst = fact_anal_dmp_osc_conv(t, fwhm, tau, period, phase, irf='g', intensity=expt, eps=eps)
        cond = np.linalg.norm(c_ref-c_tst)/(np.linalg.norm(c_ref)) < 1e-2
        self.assertEqual(cond, True)

    def test_fact_anal_sum_exp_dmp_osc_conv(self):
        fwhm = 0.15
        c_ref_decay = np.array([1, -0.5, -0.25])
        c_ref_osc = np.array([0.15, -0.075])
        tau = np.array([0.5, 10])
        tau_osc = np.array([0.3, 1])
        period_osc = np.array([0.5, 1])
        phase_osc = np.array([np.pi/3, np.pi/4])
        t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
        model = sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, phase_osc, 
        c_ref_decay, c_ref_osc, base=True, irf='g')
        noise = np.random.normal(0, np.max(np.abs(model))/100, model.size)
        eps = np.max(np.abs(model))/100*np.ones_like(noise)
        expt = model+noise
        c_tst_decay, c_tst_osc = \
            fact_anal_sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, phase_osc, base=True,
            irf='g', intensity=expt, eps=eps)
        cond_decay = np.linalg.norm(c_ref_decay-c_tst_decay)/(np.linalg.norm(c_ref_decay)) < 1e-2
        cond_osc = np.linalg.norm(c_ref_osc-c_tst_osc)/(np.linalg.norm(c_ref_osc)) < 1e-1
        self.assertEqual((cond_decay, cond_osc), (True, True))


if __name__ == '__main__':
    unittest.main()

