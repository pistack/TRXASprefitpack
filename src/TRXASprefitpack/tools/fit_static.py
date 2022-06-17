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
from ..mathfun import gen_theory_data
from .misc import read_data, plot_result
from lmfit import Parameters, fit_report, minimize


def fit_static():

    def magick(e, thy_data, no_base, data=None, eps=None):
        if no_base:
            A = thy_data/eps
            y = data/eps
            c, _, _, _ = LA.lstsq(A.reshape(A.size, 1), y)
            c = np.hstack((c,np.zeros(2)))
        else:
            A = np.ones((e.shape[0], 3))
            A[:, 0] = thy_data
            A[:, 1] = e
            A[:, 0] = A[:, 0]/eps
            A[:, 1] = A[:, 1]/eps
            A[:, 2] = A[:, 2]/eps
            y = data/eps
            c, _, _, _ = LA.lstsq(A, y)
        return c

    def residual(params, e, peaks, policy, no_base, data=None, eps=None):
        fwhm_G = params['fwhm_G']
        fwhm_L = params['fwhm_L']
        peak_factor = params['peak_factor']
        chi = np.zeros(data.shape)
        thy_data = gen_theory_data(e, peaks, 1, fwhm_G, fwhm_L, peak_factor, policy)
        for i in range(data.shape[1]):
            c = magick(e, thy_data, no_base,
                       data=data[:, i], eps=eps[:, i])
            chi[:, i] = data[:, i] - (c[0]*thy_data+c[1]*e+c[2])
        chi = chi.flatten()/eps.flatten()
        return chi

    description = '''
fit static: fitting static spectrum with theoretically calculated line spectrum
broadened by spectral line shape
v: voigt profile,
g: gaussian,
l: lorenzian,
It uses lmfit python package to fit experimental spectrum and estimates the error bound of
broadening and peak parameter
Moreover, it uses linear baseline to correct baseline feature of experimental spectrum
'''

    epilog = ''' 
*Note
energy unit for measured static spectrum and theoretically calculated spectrum should be same
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
    parse.add_argument('--fix_fwhm_G', action='store_true',
    help='fix gaussian fwhm value')
    parse.add_argument('--fix_fwhm_L', action='store_true',
    help='fix lorenzian fwhm value')
    parse.add_argument('--slow', action='store_true',
    help='use slower but robust global optimization algorithm')

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
    num_escan_pts = e.shape[0]
    data, eps = read_data(prefix, num_scan, num_escan_pts, 1000)

    peaks = np.genfromtxt(peak_file)

    fit_params = Parameters()
    if option == 'v':
        fit_params.add('fwhm_G', value=fwhm_lst[0], min=fwhm_lst[0]/2, max=2*fwhm_lst[0],
        vary=(not args.fix_fwhm_G))
        fit_params.add('fwhm_L', value=fwhm_lst[1], min=fwhm_lst[1]/2, max=2*fwhm_lst[1],
        vary=(not args.fix_fwhm_L))
    elif option == 'g':
        fit_params.add('fwhm_G', value=fwhm, min=fwhm/2, max=2*fwhm, vary=(not args.fix_fwhm_G))
        fit_params.add('fwhm_L', value=0, vary=False)
    else:
        fit_params.add('fwhm_G', value=0, vary=False)
        fit_params.add('fwhm_L', value=fwhm, min=fwhm/2, max=2*fwhm, vary=(not args.fix_fwhm_L))

    fit_params.add('peak_factor', value=peak_factor,
                   min=peak_factor/2,
                   max=2*peak_factor)

    # First, Nelder-Mead (if slow use ampgo instead)
    if args.slow:
        out = minimize(residual, fit_params, method='ampgo', calc_covar=False,
        args=(e, peaks, policy, no_base),
        kws={'data': data, 'eps': eps})
    else:
        out = minimize(residual, fit_params, method='Nelder', calc_covar=False,
        args=(e, peaks, policy, no_base),
        kws={'data': data, 'eps': eps})
    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(e, peaks, policy, no_base),
                   kws={'data': data, 'eps': eps})

    print(fit_report(out))
    chi2_ind = residual(out.params, e, peaks, policy, no_base, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-6)

    fwhm_G = out.params['fwhm_G']
    fwhm_L = out.params['fwhm_L']
    peak_factor = out.params['peak_factor']
    base = np.zeros((data.shape[0], data.shape[1]+1))
    fit = np.zeros((data.shape[0], data.shape[1]+1))
    base[:, 0] = e
    fit[:, 0] = e
    thy_data_opt = gen_theory_data(e, peaks, 1, fwhm_G, fwhm_L, peak_factor, policy)
    A = np.zeros(num_scan)
    for i in range(num_scan):
        c = magick(e, thy_data_opt, no_base, data=data[:, i],
                   eps=eps[:, i])
        base[:, i+1] = c[1]*e+c[2]
        fit[:, i+1] = c[0]*thy_data_opt+c[1]*e+c[2]
        A[i] = c[0]

    f = open(out_prefix+'_fit_report.txt', 'w')
    f.write(fit_report(out))
    f.close()

    np.savetxt(out_prefix+'_base.txt', base)
    np.savetxt(out_prefix+'_fit.txt', fit)
    np.savetxt(out_prefix+'_A.txt', A)

    reduced_fit = np.zeros_like(fit)
    reduced_fit[:, 0] = fit[:, 0]
    reduced_fit[:, 1:] = fit[:, 1:] - base[:, 1:]

    plot_result('static', num_scan, chi2_ind, data-base[:, 1:], eps, reduced_fit)

    return
