# fit eq
# fitting tscan data
# Using convolution of lower triangular 1st order rate equation model and
# normalized gaussian distribution
# normalized cauchy distribution
# normalized pseudo voigt profile
# (Mixing parameter eta is fixed according to
#  Journal of Applied Crystallography. 33 (6): 1311–1316.)

import argparse
import numpy as np
from ..mathfun.rate_eq import compute_signal_irf, fact_anal_model
from ..mathfun import solve_l_model
from .misc import parse_matrix
from .misc import set_bound_tau, read_data, contribution_table, plot_result
from lmfit import Parameters, fit_report, minimize

description = '''
fit eq: fitting tscan data using the solution of lower triangular 1st order rate equation covolved with gaussian/cauchy(lorenzian)/pseudo voigt irf function.
It uses lmfit python module to fitting experimental time trace data to sequential decay module.
To find contribution of each excited state species, it solves linear least square problem via scipy lstsq module.


In rate equation model, the ground state would be
1. ''first_and_last'' species
2. ''first'' species
3. ''last'' species
4. ground state is not included in the rate equation model
'''

epilog = '''
*Note
1. The number of time zero parameter should be same as the total number of scan to fit.
2. Every scan file whose prefix of filename is same should have same scan range
3. The rate equation matrix shoule be lower triangular.
4. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
'''

gs_help = '''
Index of ground state species.
1. ``first_and_last``, first and last species are both ground state
2. ``first``, first species is ground state
3. ``last``,  last species is ground state
4. Did not set., There is no ground state species in model equation.
'''

rate_eq_mat_help = '''
Filename for user supplied rate equation matrix. 
i th rate constant should be denoted by ki in rate equation matrix file.
Moreover rate equation matrix should be lower triangular.
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



def fit_eq():

    def residual(params, t, prefix, num_tau, mat_str, exclude, irf, data=None, eps=None):
        if irf in ['g', 'c']:
            fwhm = params['fwhm']
        else:
            fwhm = np.array([params['fwhm_G'], params['fwhm_L']])
        tau = np.empty(num_tau)
        for i in range(num_tau):
            tau[i] = params[f'tau_{i+1}']
        
        L = parse_matrix(mat_str, tau)
        y0 = np.zeros(L.shape[0]); y0[0] = 1
        eigval, V, c = solve_l_model(L, y0)

        sum = 0
        for i in range(prefix.size):
            sum = sum + data[i].size
        chi = np.empty(sum)
        end = 0
        for i in range(prefix.size):
            for j in range(data[i].shape[1]):
                t0 = params[f't_0_{prefix[i]}_{j+1}']
                model = compute_signal_irf(t[i]-t0, eigval, V, c, fwhm, irf)
                abs = fact_anal_model(model, exclude, data[i][:,j], eps[i][:,j])
                chi[end:end+data[i].shape[0]] = (data[i][:, j] - (abs @ model))/eps[i][:,j]
                end = end + data[i].shape[0]
        return chi

    tmp = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=tmp,
                                     description=description,
                                     epilog=epilog)
    parser.add_argument('-re_mat', '--rate_eq_mat', type=str, 
                        help=rate_eq_mat_help)
    parser.add_argument('-gsi', '--gs_index', default=None, type=str,
    choices=['first', 'last', 'first_and_last'], help=gs_help)
    parser.add_argument('--irf', default='g', choices=['g', 'c', 'pv'],
                        help=irf_help)
    parser.add_argument('--fwhm_G', type=float,
                        help=fwhm_G_help)
    parser.add_argument('--fwhm_L', type=float,
                        help=fwhm_L_help)
    parser.add_argument('prefix',
                        help='prefix for tscan files ' +
                        'It will read prefix_i.txt')
    parser.add_argument('--num_file', type=int, nargs='+',
                         help='number of scan file corresponding to each prefix')
    parser.add_argument('-t0', '--time_zeros', type=float, nargs='+',
                        help='time zeros for each tscan')
    parser.add_argument('-t0f', '--time_zeros_file',
                        help='filename for time zeros of each tscan')
    parser.add_argument('--tau', type=float, nargs='*',
                        help='lifetime of each decay path')
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

    rate_eq_mat_str = np.genfromtxt(args.rate_eq_mat, dtype=str)
    exclude = args.gs_index

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
        num_comp = rate_eq_mat_str.shape[0]
        if exclude == 'first_and_last':
            num_ex = num_comp-2
        elif exclude in ['first', 'last']:
            num_ex = num_comp-1
        else:
            num_ex = num_comp
    
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
        args=(t, prefix, num_tau, rate_eq_mat_str, exclude, irf),
        kws={'data': data, 'eps': eps})
    else:
        opt = minimize(residual, fit_params, method='nelder', calc_covar=False,
        args=(t, prefix, num_tau, rate_eq_mat_str, exclude, irf),
        kws={'data': data, 'eps': eps})

    # Then do Levenberg-Marquardt
    opt = minimize(residual, opt.params,
                   args=(t, prefix, num_tau, rate_eq_mat_str, exclude, irf),
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

    tau_opt = np.zeros(num_tau)
    for j in range(num_tau):
        tau_opt[j] = opt.params[f'tau_{j+1}']
    
    L_opt = parse_matrix(rate_eq_mat_str, tau_opt)

    y0 = np.zeros(L_opt.shape[0])
    y0[0] = 1
    
    eigval_opt, V_opt, c_opt = solve_l_model(L_opt, y0)

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
            model = compute_signal_irf(t[i]-opt.params[f't_0_{prefix[i]}_{j+1}'],
            eigval_opt, V_opt, c_opt, fwhm_opt, irf)
            abs_tmp = fact_anal_model(model, exclude, data[i][:, j], eps[i][:, j])
            fit[i][:, j+1] = abs_tmp @ model
            if exclude == 'first_and_last':
                abs[i][:, j] = abs_tmp[1:-1]
            elif exclude == 'last':
                abs[i][:, j] = abs_tmp[:-1]
            elif exclude == 'first':
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
