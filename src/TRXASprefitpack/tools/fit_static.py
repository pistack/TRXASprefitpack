# fit static
# fitting static spectrum
# with tddft calc peaks convolved by
# option
# v: voigt profile
# g: gaussian
# l: lorenzian
# to correct baseline I use linear line as baseline

import argparse
import numpy as np
import scipy.linalg as LA
from ..thy import gen_theory_data
from lmfit import Parameters, fit_report, minimize
import matplotlib.pyplot as plt


def fit_static():

    def magick(e, fwhm_G, fwhm_L, peak_factor, policy, no_base, data=None, eps=None):
        if no_base:
            A = gen_theory_data(e, peaks, 1, fwhm_G, fwhm_L, peak_factor, policy)
            A = A/eps
            y = data/eps
            c, _, _, _ = LA.lstsq(A.reshape(A.size, 1), y)
            c = np.vstack((c,np.zeros(2)))
        else:
            A = np.ones((e.shape[0], 3))
            A[:, 0] = gen_theory_data(e, peaks, 1, fwhm_G, fwhm_L, peak_factor, policy)
            A[:, 1] = e
            A[:, 0] = A[:, 0]/eps
            A[:, 1] = A[:, 1]/eps
            A[:, 2] = A[:, 2]/eps
            y = data/eps
            c, _, _, _ = LA.lstsq(A, y)
        return c

    def peak_fit(e, fwhm_G, fwhm_L, peak_factor, policy, c):
        return gen_theory_data(e, peaks, c[0], fwhm_G, fwhm_L,
                               peak_factor, policy) + c[1]*e+c[2]

    def residual(params, e, policy, no_base, data=None, eps=None):
        fwhm_G = params['fwhm_G']
        fwhm_L = params['fwhm_L']
        peak_factor = params['peak_factor']
        chi = np.zeros(data.shape)
        for i in range(data.shape[1]):
            c = magick(e, fwhm_G, fwhm_L, peak_factor, policy, no_base,
                       data=data[:, i], eps=eps[:, i])
            chi[:, i] = data[:, i] - peak_fit(e, fwhm_G, fwhm_L, peak_factor, policy, c)
        chi = chi.flatten()/eps.flatten()
        return chi

    description = '''
fit static: fitting static spectrum with theoretically calculated line spectrum
broadened by spectral line shape
v: voigt profile,
g: gaussian,
l: lorenzian,
to correct baseline, it uses linear line
'''

    epilog = ''' 
*Note
energy unit for measured static spectrum and
calc spectrum should be same
'''
    tmp = argparse.RawDescriptionHelpFormatter
    parse = argparse.ArgumentParser(formatter_class=tmp,
                                    description=description,
                                    epilog=epilog)
    parse.add_argument('-ls', '--line_shape',
                       default='v', choices=['v', 'g', 'l'],
                       help="line shape of spectrum"+'\n' +
                       "v: voigt profile" + '\n' +
                       "g: gaussian shape" + '\n' +
                       "l: lorenzian shape")
    parse.add_argument('--fwhm_G', type=float,
                        help='full width at half maximum for gaussian shape ' +
                        'It would be not used when you set lorenzian line shape')
    parse.add_argument('--fwhm_L', type=float,
                        help='full width at half maximum for lorenzian shape ' +
                        'It would be not used when you use gaussian line shape')
    parse.add_argument('--no_base', action='store_true',
    help ='Do not include linear base line during fitting process')
    parse.add_argument('--scale_energy', action='store_true',
    help='Scaling the energy of peak instead of shifting to match experimental spectrum')
    parse.add_argument('prefix',
                       help='prefix for experimental spectrum files' +
                       '\n' + 'It will read prefix_i.txt files')
    parse.add_argument('num_scan', type=int,
                       help='the number of static peak scan files')
    parse.add_argument('peak_file',
                       help='filename for theoretical line shape spectrum')
    parse.add_argument('peak_factor', type=float,
    help='parameter to match descrepency between thoretical spectrum and experimental spectrum')
    parse.add_argument('-o', '--out', help='prefix for output files')

    args = parse.parse_args()

    prefix = args.prefix
    option = args.line_shape
    num_scan = args.num_scan
    peak_file = args.peak_file
    peak_factor = args.peak_factor

    if args.out is None:
        out_prefix = prefix
    else:
        out_prefix = args.out

    if option == 'g':
        if args.fwhm_G is None:
            print("Please set fwhm_G of gaussian line shape")
            return
        else:
            fwhm = args.fwhm_G
    elif option == 'l':
        if args.fwhm_L is None:
            print("Please set fwhm_L of lorenzian line shape")
            return
        else:
            fwhm = args.fwhm_L
    else:
        if (args.fwhm_G is None) or (args.fwhm_L is None):
            print("Please set both fwhm_G and fwhm_L for Voigt line shape")
            return
        else:
            fwhm_lst = [args.fwhm_G, args.fwhm_L] 
    
    if args.scale_energy:
        policy = 'scale'
    else:
        policy = 'shift'

    no_base = args.no_base 

    e = np.genfromtxt(f'{prefix}_1.txt')[:, 0]
    data = np.zeros((e.shape[0], num_scan))
    eps = np.zeros((e.shape[0], num_scan))

    for i in range(num_scan):
        data[:, i] = np.genfromtxt(f'{prefix}_{i+1}.txt')[:, 1]
        eps[:, i] = np.genfromtxt(f'{prefix}_{i+1}.txt')[:, 2]

    peaks = np.genfromtxt(peak_file)

    fit_params = Parameters()
    if option == 'v':
        fit_params.add('fwhm_G', value=fwhm_lst[0], min=fwhm_lst[0]/2, max=2*fwhm_lst[0])
        fit_params.add('fwhm_L', value=fwhm[1], min=fwhm[1]/2, max=2*fwhm[1])
    elif option == 'g':
        fit_params.add('fwhm_G', value=1, min=fwhm_G/2, max=2*fwhm_G)
        fit_params.add('fwhm_L', value=0, vary=False)
    else:
        fit_params.add('fwhm_G', value=0, vary=False)
        fit_params.add('fwhm_L', value=1, min=fwhm_L/2, max=2*fwhm_L)

    fit_params.add('peak_factor', value=peak_factor,
                   min=peak_factor/2,
                   max=2*peak_factor)

    # First, Nelder-Mead
    out = minimize(residual, fit_params, method='nelder',
                   args=(e, policy),
                   kws={'data': data, 'eps': eps})
    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(e, policy),
                   kws={'data': data, 'eps': eps})

    print(fit_report(out))
    chi2_ind = residual(out.params, e, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-6)

    fwhm_G = out.params['fwhm_G']
    fwhm_L = out.params['fwhm_L']
    peak_factor = out.params['peak_factor']
    base = np.zeros((data.shape[0], data.shape[1]+1))
    fit = np.zeros((data.shape[0], data.shape[1]+1))
    base[:, 0] = e
    fit[:, 0] = e
    A = np.zeros(num_scan)
    for i in range(num_scan):
        c = magick(e, fwhm_G, fwhm_L, peak_factor, policy, no_base, data=data[:, i],
                   eps=eps[:, i])
        base[:, i+1] = c[1]*e+c[2]
        fit[:, i+1] = peak_fit(e, fwhm_G, fwhm_L, peak_factor, policy, c)
        A[i] = c[0]

    for i in range(num_scan):
        plt.figure(i+1)
        plt.title(f'Chi squared: {chi2_ind[i]:.2f}')
        plt.errorbar(e, data[:, i]-base[:, i+1],
                     eps[:, i], marker='o', mfc='none',
                     label=f'expt-base static {i+1}',
                     linestyle='none')
        plt.plot(e, fit[:, i+1]-base[:, i+1],
                 label=f'fit-base static {i+1}')
        plt.legend()
    plt.show()

    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_report(out))
    f.close()

    np.savetxt(out_prefix+'_base.txt', base)
    np.savetxt(out_prefix+'_fit.txt', fit)
    np.savetxt(out_prefix+'_A.txt', A)

    return
