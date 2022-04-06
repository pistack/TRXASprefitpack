# fit tscan
# fitting tscan data
# Using sum of exponential decay convolved with
# normalized gaussian distribution
# normalized cauchy distribution
# normalized pseudo voigt profile
# (Default value for eta is guessed according to
#  Journal of Applied Crystallography. 33 (6): 1311–1316.)

import argparse
import numpy as np
from ..mathfun import model_n_comp_conv, fact_anal_exp_conv
from lmfit import Parameters, fit_report, minimize
import matplotlib.pyplot as plt


def fit_tscan():

    def set_bound_tau(tau):
        bound = [tau/2, 1]
        if 0.1 < tau <= 10:
            bound = [0.05, 100]
        elif 10 < tau <= 100:
            bound = [5, 500]
        elif 100 < tau <= 1000:
            bound = [50, 2000]
        elif 1000 < tau <= 5000:
            bound = [500, 10000]
        elif 5000 < tau <= 50000:
            bound = [2500, 100000]
        elif 50000 < tau <= 500000:
            bound = [25000, 1000000]
        elif 500000 < tau <= 1000000:
            bound = [250000, 2000000]
        elif 1000000 < tau:
            bound = [tau/2, np.inf]
        return bound

    def residual(params, t, num_comp, base, irf, data=None, eps=None):
        if irf in ['g', 'c']:
            fwhm = params['fwhm']
        else:
            fwhm = np.array([params['fwhm_G'], params['fwhm_L']])
        tau = np.zeros(num_comp)
        for i in range(num_comp):
            tau[i] = params[f'tau_{i+1}']
        chi = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[1]):
            t0 = params[f't_0_{i+1}']
            c = fact_anal_exp_conv(t-t0, fwhm, tau, irf=irf,
                                   data=data[:, i], eps=eps[:, i], base=base)

            chi[:, i] = data[:, i] - \
                model_n_comp_conv(t-t0, fwhm, tau, c, base=base,
                                  irf=irf)
        chi = chi.flatten()/eps.flatten()

        return chi

    description = 'fit tscan: fitting tscan data ' + \
        'using sum of exponential decay covolved with ' + \
        'gaussian/cauchy(lorenzian)/pseudo voigt irf function ' + \
        'it uses fact_anal_exp_conv to determine best ' + \
        'c_i\'s when timezero, fwhm, ' + \
        'and time constants are given. ' + \
        'So, to use this script what you need to ' + \
        'give are only timezero, fwhm, and time constants ' + \
        'To set boundary of each parameter for fitting, ' + \
        'I use following scheme.' + '\n'*2 + \
        '''
*fwhm: temporal width of x-ray pulse
lower bound: 0.5*fwhm_init
upper bound: 2*fwhm_init

*t_0: timezero for each scan
lower bound: t_0 - 2*fwhm_init
upper bound: t_0 + 2*fwhm_init

*tau: life_time of each component
if tau < 0.1
lower bound: tau/2
upper bound: 1

if 0.1 < tau < 10
lower bound: 0.05
upper bound: 100

if 10 < tau < 100
lower bound: 5
upper bound: 500

if 100 < tau < 1000
lower bound: 50
upper bound: 2000

if 1000 < tau < 5000 then
lower bound: 500
upper bound: 10000

if 5000 < tau < 50000 then
lower bound: 2500
upper bound: 100000

if 50000 < tau < 500000 then
lower bound: 25000
upper bound: 1000000

if 500000 < tau < 1000000 then
lower bound: 250000
upper bound: 2000000

if 1000000 < tau then
lower bound: tau/2
upper bound: np.inf
'''

    epilog = '''
*Note

1. if you set shape of irf to pseudo voigt (pv), then
   you should provide two full width at half maximum
   value for gaussian and cauchy parts, respectively.

2. If you did not set tau then it assume you finds the
   timezero of this scan. So, --no_base option is discouraged.
'''
    tmp = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=tmp,
                                     description=description,
                                     epilog=epilog)
    parser.add_argument('--irf', default='g', choices=['g', 'c', 'pv'],
                        help='shape of instrument response function\n' +
                        'g: gaussian distribution\n' +
                        'c: cauchy distribution\n' +
                        'pv: pseudo voigt profile, ' +
                        'linear combination of gaussian distribution and ' +
                        'cauchy distribution ' + 'pv = eta*c+(1-eta)*g ' +
                        'the mixing parameter is guessed according to ' +
                        'Journal of Applied Crystallography. ' +
                        '33 (6): 1311–1316. ')
    parser.add_argument('--fwhm_G', type=float,
                        help='full width at half maximum for gaussian shape ' +
                        'It should not used when you set cauchy irf function')
    parser.add_argument('--fwhm_L', type=float,
                        help='full width at half maximum for cauchy shape ' +
                        'It should not used when you did not set irf or ' +
                        'use gaussian irf function')
    parser.add_argument('prefix',
                        help='prefix for tscan files ' +
                        'It will read prefix_i.txt')
    parser.add_argument('-t0', '--time_zeros', type=float, nargs='+',
                        help='time zeros for each tscan')
    parser.add_argument('-t0f', '--time_zeros_file',
                        help='filename for time zeros of each tscan')
    parser.add_argument('--tau', type=float, nargs='*',
                        help='lifetime of each component')
    parser.add_argument('--no_base', action='store_false',
                        help='exclude baseline for fitting')
    parser.add_argument('-o', '--out', default='out',
                        help='prefix for output files')
    args = parser.parse_args()

    prefix = args.prefix
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
        num_comp = tau.shape[0]

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
    data = np.zeros((t.shape[0], num_scan))
    eps = np.zeros((t.shape[0], num_scan))

    for i in range(num_scan):
        data[:, i] = np.genfromtxt(f'{prefix}_{i+1}.txt')[:, 1]
        eps[:, i] = np.genfromtxt(f'{prefix}_{i+1}.txt')[:, 2]

    print(f'fitting with {data.shape[1]} data set!\n')
    fit_params = Parameters()
    if irf in ['g', 'c']:
        fit_params.add('fwhm', value=fwhm,
                       min=0.5*fwhm, max=2*fwhm)
    elif irf == 'pv':
        fit_params.add('fwhm_G', value=args.fwhm_G,
                       min=0.5*args.fwhm_G, max=2*args.fwhm_G)
        fit_params.add('fwhm_L', value=args.fwhm_L,
                       min=0.5*args.fwhm_L, max=2*args.fwhm_L)
    for i in range(num_scan):
        fit_params.add(f't_0_{i+1}', value=time_zeros[i],
                       min=time_zeros[i]-2*fwhm,
                       max=time_zeros[i]+2*fwhm)

    if not find_zero:
        for i in range(num_comp):
            bd = set_bound_tau(tau[i])
            fit_params.add(f'tau_{i+1}', value=tau[i], min=bd[0],
                           max=bd[1])

    # Second initial guess using Nelder-Mead Method
    out = minimize(residual, fit_params, method='nelder',
                   args=(t, num_comp, base, irf),
                   kws={'data': data, 'eps': eps})

    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(t, num_comp, base),
                   kws={'data': data, 'eps': eps, 'irf': irf})

    print(fit_report(out))
    chi2_ind = residual(out.params, t, num_comp, base,
                        irf, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-len(out.params))

    fit = np.zeros((data.shape[0], data.shape[1]+1))
    fit[:, 0] = t
    tau_opt = np.zeros(num_comp)
    for j in range(num_comp):
        tau_opt[j] = out.params[f'tau_{j+1}']
    if base:
        c = np.zeros((num_comp+1, num_scan))
    else:
        c = np.zeros((num_comp, num_scan))
    for i in range(num_scan):
        if irf in ['g', 'c']:
            fwhm_out = out.params['fwhm']
        else:
            tmp_G = out.params['fwhm_G']
            tmp_L = out.params['fwhm_L']
            fwhm_out = np.array([tmp_G, tmp_L])
        c[:, i] = fact_anal_exp_conv(t-out.params[f't_0_{i+1}'],
                                     fwhm_out,
                                     tau_opt,
                                     data=data[:, i],
                                     eps=eps[:, i],
                                     base=base,
                                     irf=irf).flatten()
        fit[:, i+1] = model_n_comp_conv(t-out.params[f't_0_{i+1}'],
                                        fwhm_out,
                                        tau_opt,
                                        c[:, i],
                                        base=base,
                                        irf=irf)
    
    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_report(out))
    f.close()

    np.savetxt(out_prefix+'_fit.txt', fit)
    np.savetxt(out_prefix+'_c.txt', c)

    for i in range(num_scan):
        plt.figure(i+1)
        title = f'Chi squared: {chi2_ind[i]:.2f}'
        if find_zero:
            t0 = out.params[f't_0_{i+1}']
            title = f'time_zero: {t0.value:.4e}\n' + title
        plt.title(title)
        plt.errorbar(t, data[:, i],
                     eps[:, i], marker='o', mfc='none',
                     label=f'tscan expt {i+1}',
                     linestyle='none')
        plt.plot(t, fit[:, i+1],
                 label=f'fit tscan {i+1}')
        plt.legend()
        plt.figure(num_scan+i+1)
        title = f'Componant Plot for figure {i+1}'
        plt.title(title)
        if base:
            plt.plot(t, model_n_comp_conv(t-out.params[f't_0_{i+1}'],
            fwhm_out, np.array([1]), np.array([0, c[-1, i]]), base=True, irf=irf), label='base')
        for j in range(num_comp):
            plt.plot(t, model_n_comp_conv(t-out.params[f't_0_{i+1}'],
            fwhm_out, np.array([tau_opt[j]]), 
            np.array([c[j, i]]), base=False, irf=irf), label=f'{j+1} th component')
        plt.legend()
    plt.show()

    return
