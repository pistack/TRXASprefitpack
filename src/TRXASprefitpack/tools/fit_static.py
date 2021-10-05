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

    def magick(e, fwhm_G, fwhm_L, peak_shift, data=None, eps=None):
        A = np.ones((e.shape[0], 3))
        A[:, 0] = gen_theory_data(e, peaks, 1/1000, fwhm_G/1000, fwhm_L/1000,
                                  peak_shift/1000)
        A[:, 1] = e

        A[:, 0] = A[:, 0]/eps
        A[:, 1] = A[:, 1]/eps
        A[:, 2] = A[:, 2]/eps
        y = data/eps
        c, _, _, _ = LA.lstsq(A, y)
        return c

    def peak_fit(e, fwhm_G, fwhm_L, peak_shift, c):
        return gen_theory_data(e, peaks, c[0]/1000, fwhm_G/1000, fwhm_L/1000,
                               peak_shift/1000) + c[1]*e+c[2]

    def residual(params, e, data=None, eps=None):
        fwhm_G = params['fwhm_G']
        fwhm_L = params['fwhm_L']
        peak_shift = params['peak_shift']
        chi = np.zeros(data.shape)
        for i in range(data.shape[1]):
            c = magick(e, fwhm_G, fwhm_L, peak_shift,
                       data=data[:, i], eps=eps[:, i])
            chi[:, i] = data[:, i] - peak_fit(e, fwhm_G, fwhm_L, peak_shift, c)
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
energy unit for measured static spectrum must be KeV
and calc static spectrum must be eV
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
    parse.add_argument('prefix',
                       help='prefix for experimental static peak files' +
                       '\n' + 'It will read prefix_i.txt files')
    parse.add_argument('num_scan', type=int,
                       help='the number of static peak scan files')
    parse.add_argument('peak_name',
                       help='filename for theoretical line shape spectrum')
    parse.add_argument('-o', '--out', default='out',
                       help='prefix for output files')

    args = parse.parse_args()

    prefix = args.prefix
    option = args.line_shape
    num_scan = args.num_scan
    peak_name = args.peak_name
    out_prefix = args.out

    e = np.genfromtxt(f'{prefix}_1.txt')[:, 0]
    data = np.zeros((e.shape[0], num_scan))
    eps = np.zeros((e.shape[0], num_scan))

    for i in range(num_scan):
        data[:, i] = np.genfromtxt(f'{prefix}_{i+1}.txt')[:, 1]
        eps[:, i] = np.genfromtxt(f'{prefix}_{i+1}.txt')[:, 2]

    peaks = np.genfromtxt(peak_name)
    peaks[:, 0] = 1/1000*peaks[:, 0]
    pmax = np.amax(peaks[:, 1])
    pmax_e = peaks[(peaks[:, 1] == pmax), 0][0]
    a = (data[0, 0]-data[-1, 0])/(e[0]-e[-1])
    b = (-e[-1]*data[0, 0]+e[0]*data[-1, 0])/(e[0]-e[-1])
    data_corr = data[:, 0] - (a*e+b)
    dmax = np.amax(data_corr)
    dmax_e = e[data_corr == dmax][0]
    peak_shift = (pmax_e - dmax_e)*1000

    fit_params = Parameters()
    if option == 'v':
        fit_params.add('fwhm_G', value=1, min=0.1, max=10)
        fit_params.add('fwhm_L', value=1, min=0.1, max=10)
    elif option == 'g':
        fit_params.add('fwhm_G', value=1, min=0.1, max=10)
        fit_params.add('fwhm_L', value=0, vary=False)
    else:
        fit_params.add('fwhm_G', value=0, vary=False)
        fit_params.add('fwhm_L', value=1, min=0.1, max=10)

    fit_params.add('peak_shift', value=peak_shift,
                   min=-1.2*np.abs(peak_shift),
                   max=1.2*np.abs(peak_shift))

    # First, Nelder-Mead
    out = minimize(residual, fit_params, method='nelder',
                   args=(e,),
                   kws={'data': data, 'eps': eps})
    # Then do Levenberg-Marquardt
    out = minimize(residual, out.params,
                   args=(e,),
                   kws={'data': data, 'eps': eps})

    print(fit_report(out))
    chi2_ind = residual(out.params, e, data=data, eps=eps)
    chi2_ind = chi2_ind.reshape(data.shape)
    chi2_ind = np.sum(chi2_ind**2, axis=0)/(data.shape[0]-6)

    fwhm_G = out.params['fwhm_G']
    fwhm_L = out.params['fwhm_L']
    peak_shift = out.params['peak_shift']
    base = np.zeros((data.shape[0], data.shape[1]+1))
    fit = np.zeros((data.shape[0], data.shape[1]+1))
    base[:, 0] = e
    fit[:, 0] = e
    A = np.zeros(num_scan)
    for i in range(num_scan):
        c = magick(e, fwhm_G, fwhm_L, peak_shift, data=data[:, i],
                   eps=eps[:, i])
        base[:, i+1] = c[1]*e+c[2]
        fit[:, i+1] = peak_fit(e, fwhm_G, fwhm_L, peak_shift, c)
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
