# fit tscan
# fitting tscan data
# Using sum of exponential decay convolved with
# normalized gaussian distribution
# normalized cauchy distribution
# normalized pseudo voigt profile
# (Mixing parameter eta is fixed according to
#  Journal of Applied Crystallography. 33 (6): 1311–1316.)

import argparse
import numpy as np
from ..mathfun import deriv_exp_sum_conv_gau
from ..mathfun.A_matrix import make_A_matrix_exp, fact_anal_A
from .misc import set_bound_tau, read_data, contribution_table, plot_result
from lmfit import Parameters, fit_report, minimize
from scipy.optimize import minimize as opt_minimize
from ampgo import ampgo

description = '''
fit tscan: fitting experimental time trace spectrum data with the convolution of the sum of exponential decay and irf function
There are three types of irf function (gaussian, cauchy, pseudo voigt)
It uses lmfit python package to fitting time trace data and estimates error bound of irf parameter and lifetime constants. 
To calculate the contribution of each life time component, it solve least linear square problem via scipy linalg lstsq module.
'''

epilog = '''
*Note

1. The number of time zero parameter should be same as the
   total number of scan to fit.

2. Every scan file whose prefix of filename is same should have same scan range

3. if you set shape of irf to pseudo voigt (pv), then
   you should provide two full width at half maximum
   value for gaussian and cauchy parts, respectively.

4. If you did not set tau then it assume you finds the
   timezero of this scan. So, --no_base option is discouraged.
'''

irf_help = '''
shape of instrument response functon
g: gaussian distribution
c: cauchy distribution
pv: pseudo voigt profile, linear combination of gaussian distribution and cauchy distribution 
    pv = eta*c+(1-eta)*g 
    the mixing parameter is fixed according to Journal of Applied Crystallography. 33 (6): 1311–1316. 
'''

fwhm_G_help = '''
full width at half maximum for gaussian shape
It would not be used when you set cauchy irf function
'''

fwhm_L_help = '''
full width at half maximum for cauchy shape
It would not be used when you did not set irf or use gaussian irf function
'''


def fit_tscan():

    def residual(params, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        params = np.atleast_1d(params)

        if irf in ['g', 'c']:
            num_irf = 1
            fwhm = params[0]
        else:
            num_irf = 2
            fwhm = np.array([params[0], params[1]])
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
                A = make_A_matrix_exp(t[i]-t0, fwhm, tau, base, irf)
                c = fact_anal_A(A, data[i][:,j], eps[i][:,j])
                chi[end:end+data[i].shape[0]] = ((c@A) - data[i][:, j])/eps[i][:, j]
                end = end + data[i].shape[0]
                t0_idx = t0_idx + 1
        return chi

    def residual_scaler(params, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        return np.sum(residual(params, t, prefix, num_comp, base, irf, fix_irf, data, eps)**2)
    
    def df_gau(params, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        params = np.atleast_1d(params)

        fwhm = params[0]; num_irf = 1
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
            num_param = num_param-1
        df = np.zeros((num_param, sum))
        end = 0; t0_idx = 1-1*fix_irf; tau_start = num_t0 + t0_idx
        t0_idx_curr = 1
        for i in range(prefix.size):
            step = data[i].shape[0]
            for j in range(data[i].shape[1]):
                t0 = params[t0_idx_curr]

                A = make_A_matrix_exp(t[i]-t0, fwhm, tau, base, irf)
                c = fact_anal_A(A, data[i][:,j], eps[i][:,j])

                grad = deriv_exp_sum_conv_gau(t[i]-t0, fwhm, 1/tau, c, base)
                grad = np.einsum('j,ij->ij', 1/eps[i][:, j], grad)
                df[tau_start:, end:end+step] = np.einsum('i,ij->ij', -1/tau**2, grad[2:,:])
                df[t0_idx, end:end+step] = -grad[0, :]

                if not fix_irf:
                    df[0, end:end+step] = grad[1, :]
                
                end = end + step
                t0_idx = t0_idx + 1
                t0_idx_curr = t0_idx_curr + 1

        return df
    
    def grad_f_gau(params, t, prefix, num_comp, base, irf, fix_irf, data=None, eps=None):
        res = residual(params, t, prefix, num_comp, base, irf, fix_irf, data, eps)
        df = df_gau(params, t, prefix, num_comp, base, irf, fix_irf, data, eps)
        return df @ res

    tmp = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=tmp,
                                     description=description,
                                     epilog=epilog)
    parser.add_argument('--irf', default='g', choices=['g', 'c', 'pv'],
                        help=irf_help)
    parser.add_argument('--fwhm_G', type=float,
                        help=fwhm_G_help)
    parser.add_argument('--fwhm_L', type=float,
                        help=fwhm_L_help)
    parser.add_argument('prefix', nargs='+',
                        help='prefix for tscan files ' +
                        'It will read prefix_i.txt')
    parser.add_argument('--num_file', type=int, nargs='+',
                         help='number of scan file corresponding to each prefix')
    parser.add_argument('-t0', '--time_zeros', type=float, nargs='+',
                        help='time zeros for each tscan')
    parser.add_argument('-t0f', '--time_zeros_file',
                        help='filename for time zeros of each tscan')
    parser.add_argument('--tau', type=float, nargs='*',
                        help='lifetime of each component')
    parser.add_argument('--no_base', action='store_false',
                        help='exclude baseline for fitting')
    parser.add_argument('--fix_irf', action='store_true',
    help='fix irf parameter (fwhm_G, fwhm_L) during fitting process')
    parser.add_argument('--slow', action='store_true',
    help='use slower but robust global optimization algorithm')
    parser.add_argument('-o', '--out', default='out',
                        help='prefix for output files')
    args = parser.parse_args()

    prefix = np.array(args.prefix, dtype=str)
    num_file = np.array(args.num_file, dtype=int)
    out_prefix = args.out

    irf = args.irf
    if irf == 'g':
        if args.fwhm_G is None:
            print('You are using gaussian irf, so you should set fwhm_G!\n')
            return
        else:
            fwhm = args.fwhm_G
    elif irf == 'c':
        if args.fwhm_L is None:
            print('You are using cauchy/lorenzian irf,' +
                  'so you should set fwhm_L!\n')
            return
        else:
            fwhm = args.fwhm_L
    else:
        if (args.fwhm_G is None) or (args.fwhm_L is None):
            print('You are using pseudo voigt irf,' +
                  'so you should set both fwhm_G and fwhm_L!\n')
            return
        else:
            fwhm = 0.5346*args.fwhm_L + \
                np.sqrt(0.2166*args.fwhm_L**2+args.fwhm_G**2)

    if args.tau is None:
        find_zero = True  # time zero mode
        base = True
        num_comp = 0
    else:
        find_zero = False
        tau = np.array(args.tau)
        base = args.no_base
        num_comp = tau.size

    if (args.time_zeros is None) and (args.time_zeros_file is None):
        print('You should set either time_zeros or time_zeros_file!\n')
        return
    elif args.time_zeros is None:
        time_zeros = np.genfromtxt(args.time_zeros_file)
    else:
        time_zeros = np.array(args.time_zeros)

    t = np.empty(prefix.size, dtype=object)
    data = np.empty(prefix.size, dtype=object)
    eps = np.empty(prefix.size, dtype=object)
    num_scan = np.sum(num_file)

    for i in range(prefix.size):
        t[i] = np.genfromtxt(f'{prefix[i]}_1.txt')[:, 0]
        num_data_pts = t[i].size
        data[i], eps[i] = read_data(prefix[i], num_file[i], num_data_pts, 10)

    print(f'fitting with total {num_scan} data set!\n')

    fit_params = Parameters()
    if irf in ['g', 'c']:
        fit_params.add('fwhm', value=fwhm,
                       min=0.5*fwhm, max=2*fwhm, vary=(not args.fix_irf))
    elif irf == 'pv':
        fit_params.add('fwhm_G', value=args.fwhm_G,
                       min=0.5*args.fwhm_G, max=2*args.fwhm_G, vary=(not args.fix_irf))
        fit_params.add('fwhm_L', value=args.fwhm_L,
                       min=0.5*args.fwhm_L, max=2*args.fwhm_L, vary=(not args.fix_irf))

    count = 0
    for p, n in zip(prefix, num_file):
        for i in range(n):
            fit_params.add(f't_0_{p}_{i+1}', value=time_zeros[count],
            min=time_zeros[count]-2*fwhm,
            max=time_zeros[count]+2*fwhm)
            count = count + 1

    if not find_zero:
        for i in range(num_comp):
            bd = set_bound_tau(tau[i])
            fit_params.add(f'tau_{i+1}', value=tau[i], min=bd[0],
                           max=bd[1])
    
    x0 = np.empty(len(fit_params)); bd = len(fit_params)*[None]
    count = 0
    for parm in fit_params:
        x0[count] = fit_params[parm].value; bd[count] = (fit_params[parm].min, fit_params[parm].max)
        count = count+1

    # Second initial guess using global optimization algorithm
    if args.slow and irf == 'g': 
        result = ampgo(residual_scaler, bd, args=(t, prefix, num_comp, base, irf, args.fix_irf, data, eps), x0=x0, jac=grad_f_gau)
    elif args.slow and irf != 'g':
        result = ampgo(residual_scaler, bd, args=(t, prefix, num_comp, base, irf, args.fix_irf, data, eps), x0=x0)
    else:
        result = opt_minimize(residual_scaler, x0, args=(t, prefix, num_comp, base, irf, args.fix_irf, data, eps), 
        method='Nelder-Mead', bounds=bd, tol=1e-7, options={'maxfev':2000*(len(fit_params)+1)})
    
    count = 0
    for parm in fit_params:
        fit_params[parm].value = result['x'][count]
        count = count+1
    # Then do Levenberg-Marquardt
    if irf == 'g':
        opt = minimize(residual, fit_params,
        args=(t, prefix, num_comp, base, irf, args.fix_irf),
        kws={'data': data, 'eps': eps}, Dfun=df_gau, col_deriv=1)
    else:
        opt = minimize(residual, fit_params,
        args=(t, prefix, num_comp, base, irf, args.fix_irf),
        kws={'data': data, 'eps': eps})

    fit = np.empty(prefix.size, dtype=object); res = np.empty(prefix.size, dtype=object)
    for i in range(prefix.size):
        fit[i] = np.empty((data[i].shape[0], data[i].shape[1]+1))
        res[i] = np.empty((data[i].shape[0], data[i].shape[1]+1))
        fit[i][:, 0] = t[i]; res[i][:, 0] = t[i]

    if irf in ['g', 'c']:
        fwhm_opt = opt.params['fwhm']
    else:
        tmp_G = opt.params['fwhm_G']
        tmp_L = opt.params['fwhm_L']
        fwhm_opt = np.array([tmp_G, tmp_L])

    tau_opt = np.zeros(num_comp)
    for j in range(num_comp):
        tau_opt[j] = opt.params[f'tau_{j+1}']

    # Calc individual chi2
    chi = residual(opt.params, t, prefix, num_comp, base,
                        irf, args.fix_irf, data=data, eps=eps)
    
    start = 0; end = 0; chi2_ind = np.empty(prefix.size, dtype=object)
    num_param_ind = tau_opt.size+2+1*(irf == 'pv')+1*base
    for i in range(prefix.size):
        end = start + data[i].size
        chi_aux = chi[start:end].reshape(data[i].shape)
        chi2_ind_aux = np.sum(chi_aux**2, axis=0)/(data[i].shape[0]-num_param_ind)
        chi2_ind[i] = chi2_ind_aux
        start = end
    
    c = np.empty(prefix.size, dtype=object)
    for i in range(prefix.size):
        if base:
            c[i] = np.empty((num_comp+1, num_file[i]))
        else:
            c[i] = np.empty((num_comp, num_file[i]))
        
        for j in range(num_file[i]):
            A = make_A_matrix_exp(t[i]-opt.params[f't_0_{prefix[i]}_{j+1}'],
            fwhm_opt, tau_opt, base, irf)
            c_tmp = fact_anal_A(A, data[i][:, j], eps[i][:, j])
            c[i][:, j] = c_tmp
            fit[i][:, j+1] = c_tmp @ A
        res[i][:, 1:] = data[i] - fit[i][:, 1:]
    contrib_table = ''
    for i in range(prefix.size):
        contrib_table = contrib_table + '\n' + \
            contribution_table('tscan', f'Component Contribution of {prefix[i]}',
            num_file[i], num_comp, c[i])

    fit_content = fit_report(opt) + contrib_table

    print(fit_content)

    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_content)
    f.close()

    for i in range(prefix.size):
        np.savetxt(f'{out_prefix}_{prefix[i]}_fit.txt', fit[i])
        np.savetxt(f'{out_prefix}_{prefix[i]}_c.txt', c[i])

    # save residual of individual fitting 
    for i in range(prefix.size):
        for j in range(data[i].shape[1]):
            res_ind = np.vstack((res[i][:, 0], res[i][:, j+1], eps[i][:, j]))
            np.savetxt(f'{out_prefix}_{prefix[i]}_res_{j+1}.txt', res_ind.T)
    
    for i in range(prefix.size):
        plot_result(f'tscan_{prefix[i]}', num_file[i], chi2_ind[i], data[i], eps[i], fit[i], res[i])

    return
