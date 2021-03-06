# fit osc
# fitting residual of time scan data with
# convolution of sum of damped oscillation and
# normalized gaussian distribution
# normalized cauchy distribution
# normalized pseudo voigt profile
# (Mixing parameter eta is fixed according to
#  Journal of Applied Crystallography. 33 (6): 1311–1316.)

import argparse
import numpy as np
from ..mathfun.A_matrix import make_A_matrix_dmp_osc, fact_anal_A
from ..mathfun import dmp_osc_conv, fact_anal_dmp_osc_conv
from .misc import set_bound_tau, read_data, contribution_table, plot_result
from lmfit import Parameters, fit_report, minimize

description = '''
fit osc: fitting residual of experimental time trace spectrum data with the convolution of the sum of damped oscilliation and irf function
There are three types of irf function (gaussian, cauchy, pseudo voigt)
It uses lmfit python package to fitting residual of time trace data and estimates error bound of irf parameter, lifetime constants and
period of vibrations. 
To calculate the contribution of each damped oscilliation component, it solve least linear square problem via scipy linalg lstsq module.
'''

epilog = '''
*Note

1. The number of tau, period and phase parameter should be same

2. The number of time zero should be same as the number of residual scan file to fit.

3. phase should be confined in [-pi/2, pi/2] (pi ~ 3.14)

4. If you set shape of irf to pseudo voigt (pv), then
   you should provide two full width at half maximum
   value for gaussian and cauchy parts, respectively.
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


def fit_osc():

    def residual(params, t, num_comp, irf, data=None, eps=None):
        if irf in ['g', 'c']:
            fwhm = params['fwhm']
        else:
            fwhm = np.array([params['fwhm_G'], params['fwhm_L']])
        tau = np.empty(num_comp)
        T = np.empty(num_comp)
        phase = np.empty(num_comp)
        for i in range(num_comp):
            tau[i] = params[f'tau_{i+1}']
            T[i] = params[f'period_{i+1}']
            phase[i] = params[f'phi_{i+1}']
        chi = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[1]):
            t0 = params[f't_0_{i+1}']
            A = make_A_matrix_dmp_osc(t-t0, fwhm, tau, T, phase, irf)
            c = fact_anal_A(A, data[:,i], eps[:, i])
            chi[:, i] = data[:, i] - (c@A)
        chi = chi.flatten()/eps.flatten()

        return chi

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
    parser.add_argument('prefix',
                        help='prefix of residual of time delay scan data to fit' +
                        'It will prefix_i.txt')
    parser.add_argument('-t0', '--time_zeros', type=float, nargs='+',
                        help='time zeros for each residual of tscan')
    parser.add_argument('-t0f', '--time_zeros_file',
                        help='filename for time zeros of each residual of tscan')
    parser.add_argument('--tau', type=float, nargs='+',
                        help='lifetime of each component')
    parser.add_argument('--period', type=float, nargs='+',
                        help='period of the vibration of each component')
    parser.add_argument('--phase', type=float, nargs='+',
    help='phase factor of each damped oscilliation component')
    parser.add_argument('--fix_irf', action='store_true',
    help='fix irf parameter (fwhm_G, fwhm_L) during fitting process')
    parser.add_argument('--fix_t0', action='store_true',
    help='fix time zero during fitting process')
    parser.add_argument('--slow', action='store_true',
    help='use slower but robust global optimization algorithm')
    parser.add_argument('-o', '--out', default=None,
                        help='prefix for output files')
    args = parser.parse_args()

    prefix = args.prefix
    if args.out is None:
        args.out = prefix
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


    tau = np.array(args.tau)
    T = np.array(args.period)
    phi = np.array(args.phase)
    num_comp = tau.size

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
                       max=time_zeros[i]+2*fwhm, vary=(not args.fix_t0))


    for i in range(num_comp):
        bd1 = set_bound_tau(tau[i])
        bd2 = set_bound_tau(T[i])
        fit_params.add(f'tau_{i+1}', value=tau[i], min=bd1[0], max=bd1[1])
        fit_params.add(f'period_{i+1}', value=T[i], min=bd2[0], max=bd2[1])
        fit_params.add(f'phi_{i+1}', value=phi[i], min=-np.pi/2, max=np.pi/2+0.1)

    # Second initial guess using global optimization algorithm
    if args.slow: 
        out = minimize(residual, fit_params, method='ampgo', calc_covar=False,
        args=(t, num_comp, irf),
        kws={'data': data, 'eps': eps})
    else:
        out = minimize(residual, fit_params, method='nelder', calc_covar=False,
        args=(t, num_comp, irf),
        kws={'data': data, 'eps': eps})

    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(t, num_comp),
                   kws={'data': data, 'eps': eps, 'irf': irf})

    chi2_ind = residual(out.params, t, num_comp, irf, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-len(out.params))

    fit = np.zeros((data.shape[0], data.shape[1]+1))
    fit[:, 0] = t

    if irf in ['g', 'c']:
        fwhm_opt = out.params['fwhm']
    else:
        tmp_G = out.params['fwhm_G']
        tmp_L = out.params['fwhm_L']
        fwhm_opt = np.array([tmp_G, tmp_L])

    tau_opt = np.zeros(num_comp)
    T_opt = np.zeros(num_comp)
    phi_opt = np.zeros(num_comp)

    for j in range(num_comp):
        tau_opt[j] = out.params[f'tau_{j+1}']
        T_opt[j] = out.params[f'period_{j+1}']
        phi_opt[j] = out.params[f'phi_{j+1}']

    c = np.zeros((num_comp, num_scan))
    for i in range(num_scan):
        c[:, i] = fact_anal_dmp_osc_conv(t-out.params[f't_0_{i+1}'], fwhm_opt,
        tau_opt, T_opt, phi_opt, irf=irf, data=data[:, i], eps=eps[:, i])
        fit[:, i+1] = dmp_osc_conv(t-out.params[f't_0_{i+1}'], fwhm_opt,
        tau_opt, T_opt, phi_opt, c[:, i], irf=irf)
    
    contrib_table = contribution_table('tscan', 'Component Contribution',
    num_scan, num_comp, c)

    fit_content = fit_report(out) + '\n' + contrib_table

    print(fit_content)

    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_content)
    f.close()

    np.savetxt(out_prefix+'_fit.txt', fit)
    np.savetxt(out_prefix+'_c.txt', c)

    plot_result('res tscan', num_scan, chi2_ind, data, eps, fit)

    return
