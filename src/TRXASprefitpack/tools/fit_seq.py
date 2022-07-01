# fit seq
# fitting tscan data
# Using sequential decay model convolved with
# normalized gaussian distribution
# normalized cauchy distribution
# normalized pseudo voigt profile
# (Mixing parameter eta is fixed according to
#  Journal of Applied Crystallography. 33 (6): 1311–1316.)

import argparse
import numpy as np
from ..mathfun.rate_eq import compute_signal_irf, fact_anal_model
from ..mathfun import solve_seq_model, rate_eq_conv, fact_anal_rate_eq_conv
from .misc import set_bound_tau, read_data, contribution_table, plot_result
from lmfit import Parameters, fit_report, minimize

description = '''
fit seq: fitting tscan data using the solution of sequtial decay equation covolved with gaussian/cauchy(lorenzian)/pseudo voigt irf function.
It uses lmfit python module to fitting experimental time trace data to sequential decay module.
To find contribution of each excited state species, it solves linear least square problem via scipy lstsq module.


It supports 4 types of sequential decay
Type 0: both raising and decay
    GS -> 1 -> 2 -> ... -> n -> GS
Type 1: no raising
    1 -> 2 -> ... -> n -> GS
Type 2: no decay
    GS -> 1 -> 2 -> ... -> n
Type 3: Neither raising nor decay
    1 -> 2 -> ... -> n 
'''

epilog = '''
*Note
1. The number of time zero parameter should be same as the number of scan to fit.
2. Type 0 sequential decay needs n+1 lifetime constants for n excited state species
3. Type 1, 2 sequential decay needs n lifetime constants for n excited state species
4. Type 3 sequential decay needs n-1 lifetime constants for n excited state species
5. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
'''

seq_decay_help = '''
type of sequential decay
0. GS -> 1 -> 2 -> ... -> n -> GS (both raising and decay)
1. 1 -> 2 -> ... -> n -> GS (No raising)
2. GS -> 1 -> 2 -> ... -> n (No decay)
3. 1 -> 2 -> ... -> n (Neither raising nor decay)
Default option is type 0 both raising and decay

*Note
1. type 0 needs n+1 lifetime value
2. type 1 and 2 need n lifetime value
3. type 3 needs n-1 lifetime value
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



def fit_seq():

    def residual(params, t, num_tau, exclude, irf, data=None, eps=None):
        if irf in ['g', 'c']:
            fwhm = params['fwhm']
        else:
            fwhm = np.array([params['fwhm_G'], params['fwhm_L']])
        tau = np.zeros(num_tau)
        for i in range(num_tau):
            tau[i] = params[f'tau_{i+1}']
        eigval, V, c = solve_seq_model(tau)

        chi = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[1]):
            t0 = params[f't_0_{i+1}']
            model = compute_signal_irf(t-t0, eigval, V, c, fwhm, irf)
            abs = fact_anal_model(model, exclude, data[:, i], eps[:, i])
            chi[:, i] = data[:, i] - (abs @ model)
        chi = chi.flatten()/eps.flatten()

        return chi

    tmp = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=tmp,
                                     description=description,
                                     epilog=epilog)
    parser.add_argument('-sdt', '--seq_decay_type', type=int, default=0,
                        help=seq_decay_help)
    parser.add_argument('--irf', default='g', choices=['g', 'c', 'pv'],
                        help=irf_help)
    parser.add_argument('--fwhm_G', type=float,
                        help=fwhm_G_help)
    parser.add_argument('--fwhm_L', type=float,
                        help=fwhm_L_help)
    parser.add_argument('prefix',
                        help='prefix for tscan files ' +
                        'It will read prefix_i.txt')
    parser.add_argument('-t0', '--time_zeros', type=float, nargs='+',
                        help='time zeros for each tscan')
    parser.add_argument('-t0f', '--time_zeros_file',
                        help='filename for time zeros of each tscan')
    parser.add_argument('--tau', type=float, nargs='*',
                        help='lifetime of each decay')
    parser.add_argument('--fix_irf', action='store_true',
    help='fix irf parameter (fwhm_G, fwhm_L) during fitting process')
    parser.add_argument('--slow', action='store_true',
    help='use slower but robust global optimization algorithm')
    parser.add_argument('-o', '--out', default=None,
                        help='prefix for output files')
    args = parser.parse_args()

    prefix = args.prefix
    if args.out is None:
        args.out = prefix
    out_prefix = args.out

    seq_decay_type = args.seq_decay_type

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
        print('Please set lifetime constants for each decay')
        return
    else:
        tau = np.array(args.tau)
        num_tau = tau.size
        if seq_decay_type == 0:
            num_ex = num_tau-1
            exclude = 'first_and_last'
        elif seq_decay_type == 1:
            num_ex = num_tau
            exclude = 'last'
        elif seq_decay_type == 2:
            num_ex = num_tau
            exclude = 'first'
        else:
            num_ex = num_tau+1
            exclude = None
    
    if (args.time_zeros is None) and (args.time_zeros_file is None):
        print('You should set either time_zeros or time_zeros_file!\n')
        return
    elif args.time_zeros is None:
        time_zeros = np.genfromtxt(args.time_zeros_file)
        num_scan = time_zeros.size
    else:
        time_zeros = np.array(args.time_zeros)
        num_scan = time_zeros.size

    t = np.genfromtxt(f'{prefix}_1.txt')[:, 0]
    num_data_pts = t.size
    data, eps = read_data(prefix, num_scan, num_data_pts, 10)

    print(f'fitting with {num_scan} data set!\n')
    fit_params = Parameters()
    if irf in ['g', 'c']:
        fit_params.add('fwhm', value=fwhm,
                       min=0.5*fwhm, max=2*fwhm, vary=(not args.fix_irf))
    elif irf == 'pv':
        fit_params.add('fwhm_G', value=args.fwhm_G,
                       min=0.5*args.fwhm_G, max=2*args.fwhm_G, vary=(not args.fix_irf))
        fit_params.add('fwhm_L', value=args.fwhm_L,
                       min=0.5*args.fwhm_L, max=2*args.fwhm_L, vary=(not args.fix_irf))
    for i in range(num_scan):
        fit_params.add(f't_0_{i+1}', value=time_zeros[i],
                       min=time_zeros[i]-2*fwhm,
                       max=time_zeros[i]+2*fwhm)

    for i in range(num_tau):
        bd = set_bound_tau(tau[i])
        fit_params.add(f'tau_{i+1}', value=tau[i], min=bd[0], max=bd[1])

    # Second initial guess using global optimization algorithm
    if args.slow: 
        out = minimize(residual, fit_params, method='ampgo', calc_covar=False,
        args=(t, num_tau, exclude, irf),
        kws={'data': data, 'eps': eps})
    else:
        out = minimize(residual, fit_params, method='nelder', calc_covar=False,
        args=(t, num_tau, exclude, irf),
        kws={'data': data, 'eps': eps})

    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(t, num_tau, exclude),
                   kws={'data': data, 'eps': eps, 'irf': irf})

    chi2_ind = residual(out.params, t, num_tau, exclude,
                        irf, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-len(out.params))

    fit = np.zeros((data.shape[0], data.shape[1]+1))
    res = np.zeros((data.shape[0], data.shape[1]+1))
    fit[:, 0] = t
    res[:, 0] = t

    if irf in ['g', 'c']:
        fwhm_opt = out.params['fwhm']
    else:
        tmp_G = out.params['fwhm_G']
        tmp_L = out.params['fwhm_L']
        fwhm_opt = np.array([tmp_G, tmp_L])

    tau_opt = np.zeros(num_tau)
    for j in range(num_tau):
        tau_opt[j] = out.params[f'tau_{j+1}']
    
    eigval_opt, V_opt, c_opt = solve_seq_model(tau_opt)

    abs = np.zeros((num_ex, num_scan))
    for i in range(num_scan):
        abs_tmp = fact_anal_rate_eq_conv(t-out.params[f't_0_{i+1}'],
        fwhm_opt, eigval_opt, V_opt, c_opt, exclude, 
        data=data[:, i], eps=eps[:, i], irf=irf)
        fit[:, i+1] = rate_eq_conv(t-out.params[f't_0_{i+1}'],
        fwhm_opt, abs_tmp, eigval_opt, V_opt, c_opt, irf=irf)
        if seq_decay_type == 0:
            abs[:, i] = abs_tmp[1:-1]
        elif seq_decay_type == 1:
            abs[:, i] = abs_tmp[:-1]
        elif seq_decay_type == 2:
            abs[:, i] = abs_tmp[1:]
        else:
            abs[:, i] = abs_tmp
    
    res[:, 1:] = data - fit[:, 1:]
    
    contrib_table = contribution_table('tscan', 'Excited State Contribution', num_scan,
    num_ex, abs)
    fit_content = fit_report(out) + '\n' + contrib_table

    print(fit_content)

    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_content)
    f.close()

    np.savetxt(out_prefix+'_fit.txt', fit)
    np.savetxt(out_prefix+'_abs.txt', abs)

    # save residual of individual fitting 

    for i in range(data.shape[1]):
        res_ind = np.vstack((res[:, 0], res[:, i+1], eps[:, i]))
        np.savetxt(out_prefix+f'_res_{i+1}.txt', res_ind.T)

    plot_result('tscan', num_scan, chi2_ind, data, eps, fit, res)

    return
