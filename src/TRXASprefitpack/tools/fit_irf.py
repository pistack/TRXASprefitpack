# fit irf
# fitting irf function
# There are three irf shapes are avaliable
# normalized gaussian distribution
# normalized cauchy distribution
# normalized pseudo voigt profile
# (Mixing parameter eta is fixed according to
#  Journal of Applied Crystallography. 33 (6): 1311–1316.)

import argparse
import numpy as np
from ..mathfun import gau_irf, cauchy_irf, pvoigt_irf
from lmfit import Parameters, fit_report, minimize
import matplotlib.pyplot as plt


def fit_irf():

    def residual(params, t, irf, data=None, eps=None):
        if irf in ['g', 'c']:
            fwhm = params['fwhm']
        else:
            fwhm = np.array([params['fwhm_G'], params['fwhm_L']])
        chi = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[1]):
            t0 = params[f't_0_{i+1}']
            if irf == 'g':
                chi[:, i] = data[:, i] - gau_irf(t-t0, fwhm)
            elif irf == 'c':
                chi[:, i] = data[:, i] - cauchy_irf(t-t0, fwhm)
            else:
                f = fwhm[0]**5+2.69269*fwhm[0]**4*fwhm[1]+2.42843*fwhm[0]**3*fwhm[1]**2 + \
                    4.47163*fwhm[0]**2*fwhm[1]**3 + 0.07842*fwhm[0]*fwhm[1]**4 + fwhm[1]**5
                f = f**(1/5)
                eta = 1.36603*(fwhm[1]/f)-0.47719*(fwhm[1]/f)**2+0.11116*(fwhm[1]/f)**3
                chi[:, i] = data[:, i] - pvoigt_irf(t-t0, fwhm[0], fwhm[1], eta)
        chi = chi.flatten()/eps.flatten()

        return chi

    description = 'fit irf: fitting instrumental response function ' + \
        'There are three irf shapes are avaliable. ' + \
        '1. gaussian 2. cauchy(lorenzian) 3. pseudo voigt shape' + \
        'To set boundary of each parameter for fitting, ' + \
        'I use following scheme.' + '\n'*2 + \
        '''
*fwhm: temporal width of pulse
lower bound: 0.5*fwhm_init
upper bound: 2*fwhm_init

*t_0: timezero or center of irf function for each scan
lower bound: t_0 - 2*fwhm_init
upper bound: t_0 + 2*fwhm_init
'''

    epilog = '''
*Note

1. if you set shape of irf to pseudo voigt (pv), then
   you should provide two full width at half maximum
   value for gaussian and cauchy parts, respectively.
2. This script is only useful when one can directly measure
   instrumental response function.
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
                        'the mixing parameter is fixed according to ' +
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

    # Second initial guess using global optimization algorithm
    if args.slow: 
        out = minimize(residual, fit_params, method='ampgo',
        args=(t, irf),
        kws={'data': data, 'eps': eps})
    else:
        out = minimize(residual, fit_params, method='nelder',
        args=(t, irf),
        kws={'data': data, 'eps': eps})

    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(t, irf),
                   kws={'data': data, 'eps': eps})

    chi2_ind = residual(out.params, t, irf, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-len(out.params))

    fit = np.zeros((data.shape[0], data.shape[1]+1))
    fit[:, 0] = t

    for i in range(num_scan):
        if irf in ['g', 'c']:
            fwhm_out = out.params['fwhm']
        else:
            tmp_G = out.params['fwhm_G']
            tmp_L = out.params['fwhm_L']
            fwhm_out = np.array([tmp_G, tmp_L])
        
        if irf == 'g':
            fit[:, i+1] = gau_irf(t-out.params[f't_0_{i+1}'],
            fwhm_out)
        elif irf == 'c':
            fit[:, i+1] = cauchy_irf(t-out.params[f't_0_{i+1}'],
            fwhm_out)
        else:
            f_out = fwhm_out[0]**5+2.69269*fwhm_out[0]**4*fwhm_out[1]+2.42843*fwhm_out[0]**3*fwhm_out[1]**2 + \
                    4.47163*fwhm_out[0]**2*fwhm_out[1]**3 + 0.07842*fwhm_out[0]*fwhm_out[1]**4 + fwhm_out[1]**5
            f_out = f_out**(1/5)
            eta_out = 1.36603*(fwhm_out[1]/f_out)-0.47719*(fwhm_out[1]/f_out)**2+0.11116*(fwhm_out[1]/f_out)**3
            fit[:, i+1] = pvoigt_irf(t-out.params[f't_0_{i+1}'], fwhm_out[0], fwhm_out[1], eta_out)
    
    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_report(out))
    f.close()

    np.savetxt(out_prefix+'_fit.txt', fit)

    print(fit_report(out))


    for i in range(num_scan):
        plt.figure(i+1)
        title = f'Chi squared: {chi2_ind[i]:.2f}'
        plt.title(title)
        plt.errorbar(t, data[:, i],
                     eps[:, i], marker='o', mfc='none',
                     label=f'tscan expt {i+1}',
                     linestyle='none')
        plt.plot(t, fit[:, i+1],
                 label=f'fit tscan {i+1}')
        plt.legend()
    plt.show()

    return
