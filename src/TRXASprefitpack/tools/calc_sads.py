# calc_sads
# Calculate species associated difference spectrum from experimental 
# energy scan data and the convolution of lower triangular 1st order rate equation model and
# instrumental response function.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ..mathfun import solve_seq_model, solve_l_model, sads
from .misc import parse_matrix

description = '''
calc sads: Calculate species associated difference spectrum from experimental energy scan data and
the convolution of lower triangular 1st order rate equation model and instrumental response function

In rate equation model, the ground state would be
1. ''first_and_last'' species
2. ''first'' species
3. ''last'' species
4. ground state is not included in the rate equation model
'''

epilog = '''
*Note
1. Associated difference spectrum of ground state species is zero.
2. The rate equation matrix shoule be lower triangular.
3. Rate equation matrix for sequential decay model is sparse, so if your model is sequential decay then use --seq option
4. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
'''

gs_help = '''
Index of ground state species.
1. ``first_and_last``, first and last species are both ground state
2. ``first``, first species is ground state
3. ``last``,  last species is ground state
4. Did not set., There is no ground state species in model equation.
'''

seq_help = '''
Use sequential decay dynamics instead of more general lower triangular one.
If this option is turned on, it use following sequential decay dynamics
1 -> 2 -> 3 -> ... -> last
You can control the behavior of first and last species via --gsi option
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
    the mixing parameter is fixed according to Journal of Applied Crystallography. 33 (6): 1311???1316. 
'''

fwhm_G_help = '''
full width at half maximum for gaussian shape
It would not be used when you set cauchy irf function
'''

fwhm_L_help = '''
full width at half maximum for cauchy shape
It would not be used when you did not set irf or use gaussian irf function
'''



def calc_sads():

    tmp = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=tmp,
                                     description=description,
                                     epilog=epilog)
    parser.add_argument('-re_mat', '--rate_eq_mat', type=str, 
                        help=rate_eq_mat_help)
    parser.add_argument('--seq', action='store_true',
    help=seq_help)
    parser.add_argument('-gsi', '--gs_index', default=None, type=str, 
    choices=['first', 'last', 'first_and_last'], help=gs_help)
    parser.add_argument('--irf', default='g', choices=['g', 'c', 'pv'],
                        help=irf_help)
    parser.add_argument('--fwhm_G', type=float,
                        help=fwhm_G_help)
    parser.add_argument('--fwhm_L', type=float,
                        help=fwhm_L_help)
    parser.add_argument('escan_file',
                        help='filename for scale corrected energy scan file')
    parser.add_argument('escan_err_file',
    help='filename for the scaled estimated experimental error of energy scan file')
    parser.add_argument('-t0', '--time_zero', type=float,
                        help='time zero of energy scan')
    parser.add_argument('--escan_time', type=float, nargs='+',
    help='time delay for each energy scan')
    parser.add_argument('--tau', type=float, nargs='+',
                        help='lifetime of each decay path')
    parser.add_argument('-o', '--out', default='out',
                        help='prefix for output files')
    args = parser.parse_args()

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
            fwhm = np.array([args.fwhm_G, args.fwhm_L])

    if args.tau is None:
        print('Please set lifetime constants for each decay')
        return
    else:
        tau = np.array(args.tau)
    
    if (args.time_zero is None):
        print('You should set time_zero for energy scan \n')
        return
    else:
        time_zero = args.time_zero

    escan_data = np.genfromtxt(args.escan_file)
    escan_err = np.genfromtxt(args.escan_err_file)
    escan_time = np.array(args.escan_time)
    out_prefix = args.out
    exclude = args.gs_index

    if args.seq:
        eigval, V, c = solve_seq_model(tau)
    else:
        rate_eq_mat_str = np.genfromtxt(args.rate_eq_mat, dtype=str)
        L_mat = parse_matrix(rate_eq_mat_str, tau)
        y0 = np.zeros(L_mat.shape[0]); y0[0] = 1
        eigval, V, c = solve_l_model(L_mat, y0)
    
    ads, ads_eps = sads(escan_time-time_zero, fwhm, eigval, V, c, exclude, irf, data=escan_data[:,1:], eps=escan_err)

    e = escan_data[:,0]

    out_ads = np.vstack((e, ads)).T

    # save calculated sads results
    np.savetxt(f'{out_prefix}_sads.txt', out_ads)
    np.savetxt(f'{out_prefix}_sads_eps.txt', ads_eps.T)

    # plot sads results
    plt.title('Species Associated Difference Spectrum')
    for i in range(ads.shape[0]):
        plt.errorbar(e, ads[i,:], ads_eps[i,:], label=f'excited state {i+1}')
    plt.legend()
    plt.show()

    return