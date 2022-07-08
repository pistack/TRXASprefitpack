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
1. The number of time zero parameter should be same as the total number of scan to fit.
2. Every scan file whose prefix of filename is same should have same scan range
3. Type 0 sequential decay needs n+1 lifetime constants for n excited state species
4. Type 1, 2 sequential decay needs n lifetime constants for n excited state species
5. Type 3 sequential decay needs n-1 lifetime constants for n excited state species
6. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
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

    def residual(params, t, prefix, num_tau, exclude, irf, data=None, eps=None):
        if irf in ['g', 'c']:
            fwhm = params['fwhm']
        else:
            fwhm = np.array([params['fwhm_G'], params['fwhm_L']])
        tau = np.empty(num_tau)
        for i in range(num_tau):
            tau[i] = params[f'tau_{i+1}']
        eigval, V, c = solve_seq_model(tau)

        sum = 0
        for i in range(prefix.size):
            sum = sum + data[i].size
        chi = np.empty(sum)
        start = 0; end = 0
        for i in range(prefix.size):
            for j in range(data[i].shape[1]):
                t0 = params[f't_0_{prefix[i]}_{j+1}']
                model = compute_signal_irf(t[i]-t0, eigval, V, c, fwhm, irf)
                abs = fact_anal_model(model, exclude, data[i][:,j], eps[i][:,j])
                chi[end:end+data[i].shape[0]] = data[i][:, j] - (abs @ model)
                end = end + data[i].shape[0]
            chi[start:end] = chi[start:end]/eps[i].flatten()
            start = end
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
                        help='lifetime of each decay')
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

    for i in range(num_tau):
        bd = set_bound_tau(tau[i])
        fit_params.add(f'tau_{i+1}', value=tau[i], min=bd[0], max=bd[1])

    # Second initial guess using global optimization algorithm
    if args.slow: 
        opt = minimize(residual, fit_params, method='ampgo', calc_covar=False,
        args=(t, prefix, num_tau, exclude, irf),
        kws={'data': data, 'eps': eps})
    else:
        opt = minimize(residual, fit_params, method='nelder', calc_covar=False,
        args=(t, prefix, num_tau, exclude, irf),
        kws={'data': data, 'eps': eps})

    # Then do Levenberg-Marquardt
    opt = minimize(residual, opt.params,
                   args=(t, prefix, num_tau, exclude, irf),
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

    tau_opt = np.empty(num_tau)
    for j in range(num_tau):
        tau_opt[j] = opt.params[f'tau_{j+1}']
    
    eigval_opt, V_opt, c_opt = solve_seq_model(tau_opt)

    chi = residual(opt.params, t, prefix, num_tau, exclude, irf, data=data, eps=eps)
    start = 0; end = 0; chi2_ind = np.empty(prefix.size, dtype=object)
    num_param_ind = tau_opt.size+2+1*(irf == 'pv')
    for i in range(prefix.size):
        end = start + data[i].size
        chi_aux = chi[start:end].reshape(data[i].shape)
        chi2_ind_aux = np.sum(chi_aux**2, axis=0)/(data[i].shape[0]-num_param_ind)
        chi2_ind[i] = chi2_ind_aux
        start = end
    
    abs = np.empty(prefix.size, dtype=object)
    for i in range(prefix.size):
        abs[i] = np.zeros((num_ex, num_file[i]))
        for j in range(num_file[i]):
            abs_tmp = fact_anal_rate_eq_conv(t[i]-opt.params[f't_0_{prefix[i]}_{j+1}'],
            fwhm_opt, eigval_opt, V_opt, c_opt, exclude, data=data[i][:, j], eps=eps[i][:, j], irf=irf)
            fit[i][:, j+1] = rate_eq_conv(t[i]-opt.params[f't_0_{prefix[i]}_{j+1}'],
            fwhm_opt, abs_tmp, eigval_opt, V_opt, c_opt, irf=irf)
            if seq_decay_type == 0:
                abs[i][:, j] = abs_tmp[1:-1]
            elif seq_decay_type == 1:
                abs[i][:, j] = abs_tmp[:-1]
            elif seq_decay_type == 2:
                abs[i][:, j] = abs_tmp[1:]
            else:
                abs[i][:, j] = abs_tmp
        res[i][:, 1:] = data[i] - fit[i][:, 1:]

    contrib_table = ''
    for i in range(prefix.size):
        contrib_table = contrib_table + '\n' + \
            contribution_table('tscan', f'Excited State Contribution of {prefix[i]}',
            num_file[i], num_ex, abs[i])

    fit_content = fit_report(opt) + contrib_table

    print(fit_content)

    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_content)
    f.close()
    for i in range(prefix.size):
        np.savetxt(f'{out_prefix}_{prefix[i]}_fit.txt', fit[i])
        np.savetxt(f'{out_prefix}_{prefix[i]}_abs.txt', abs[i])

    # save residual of individual fitting 
    for i in range(prefix.size):
        for j in range(data[i].shape[1]):
            res_ind = np.vstack((res[i][:, 0], res[i][:, j+1], eps[i][:, j]))
            np.savetxt(f'{out_prefix}_{prefix[i]}_res_{j+1}.txt', res_ind.T)
    
    for i in range(prefix.size):
        plot_result(f'tscan_{prefix[i]}', num_file[i], chi2_ind[i], data[i], eps[i], fit[i], res[i])

    return
