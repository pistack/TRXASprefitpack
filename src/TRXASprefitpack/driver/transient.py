'''
transient:
submodule for fitting time delay scan with the
convolution of sum of exponential decay and instrumental response function 

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

import numpy as np
from ..mathfun.irf import calc_eta
from ..mathfun.A_matrix import make_A_matrix_exp, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy

def residual_tscan(params: np.array, t, prefix, num_comp: int, base: bool, irf: str, fix_irf: bool, data=None, eps=None):
    '''
    residual_tscan
    lmfit minimizer compatible residual function for 
    '''
        params = np.atleast_1d(params)

        if irf in ['g', 'c']:
            num_irf = 1
            fwhm = params[0]
            eta = None
        else:
            num_irf = 2
            fwhm = np.array([params[0], params[1]])
            eta = calc_eta(fwhm[0], fwhm[1])

        num_t0 = 0
        for i in range(prefix.size):
            num_t0 = data[i].shape[1] + num_t0

        tau = np.empty(num_comp, dtype=float)
        for i in range(num_comp):
            tau[i] = params[num_irf+num_t0+i]

        sum = 0
        for i in range(prefix.size):
            sum = sum + data[i].size
        chi = np.empty(sum)

        end = 0; t0_idx = num_irf
        for i in range(prefix.size):
            for j in range(data[i].shape[1]):
                t0 = params[t0_idx]
                A = make_A_matrix_exp(t[i]-t0, fwhm, tau, base, irf, eta)
                c = fact_anal_A(A, data[i][:,j], eps[i][:,j])
                chi[end:end+data[i].shape[0]] = ((c@A) - data[i][:, j])/eps[i][:, j]
                end = end + data[i].shape[0]
                t0_idx = t0_idx + 1
        return chi

    def residual_scaler(params, fwhm, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        if fix_irf:
            params = np.hstack((fwhm, params))
        return np.sum(residual(params, t, prefix, num_comp, base, irf, fix_irf, data, eps)**2)
    
    def df_accel(params, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        params = np.atleast_1d(params)

        if irf in ['g', 'c']:
            fwhm = params[0]
            num_irf = 1 
            eta = None
        else:
            fwhm = np.array([params[0], params[1]])
            num_irf = 2
            eta = calc_eta(fwhm[0], fwhm[1])

        num_t0 = 0
        for i in range(prefix.size):
            num_t0 = num_t0 + data[i].shape[1]
        
        tau = np.empty(num_comp, dtype=float)
        for i in range(num_comp):
            tau[i] = params[num_irf+num_t0+i]

        sum = 0
        for i in range(prefix.size):
            sum = sum + data[i].size

        num_param = num_irf+num_t0+num_comp

        if fix_irf:
            num_param = num_param-num_irf

        df = np.zeros((num_param, sum))

        end = 0; t0_idx = 1-1*fix_irf; tau_start = num_t0 + t0_idx
        t0_idx_curr = num_irf

        for i in range(prefix.size):
            step = data[i].shape[0]
            for j in range(data[i].shape[1]):
                t0 = params[t0_idx_curr]
                A = make_A_matrix_exp(t[i]-t0, fwhm, tau, base, irf, eta)
                c = fact_anal_A(A, data[i][:,j], eps[i][:,j])
                
                if irf == 'g':
                    grad = deriv_exp_sum_conv_gau(t[i]-t0, fwhm, 1/tau, c, base)
                elif irf == 'c':
                    grad = deriv_exp_sum_conv_cauchy(t[i]-t0, fwhm, 1/tau, c, base)
                else:
                    grad_gau = deriv_exp_sum_conv_gau(t[i]-t0, fwhm[0], 1/tau, c, base)
                    grad_cauchy = deriv_exp_sum_conv_cauchy(t[i]-t0, fwhm[1], 1/tau, c, base)
                    grad = grad_gau + eta*(grad_cauchy-grad_gau)
   
                grad = np.einsum('j,ij->ij', 1/eps[i][:, j], grad)
                df[tau_start:, end:end+step] = np.einsum('i,ij->ij', -1/tau**2, grad[2:,:])
                df[t0_idx, end:end+step] = -grad[0, :]

                if not fix_irf:
                    df[0, end:end+step] = grad[1, :]
                
                end = end + step
                t0_idx = t0_idx + 1
                t0_idx_curr = t0_idx_curr + 1

        return df
    
    def grad_f_accel(params, fwhm, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        if fix_irf:
            params = np.hstack((fwhm, params))
        res = residual(params, t, prefix, num_comp, base, irf, fix_irf, data, eps)
        df = df_accel(params, t, prefix, num_comp, base, irf, fix_irf, data, eps)
        return df @ res