# calc_dads
# Calculate decay associated difference spectrum from experimental 
# energy scan data and the convolution of sum of exponential decay and
# instrumental response function.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ..mathfun import dads
from .misc import parse_matrix

description = '''
calc dads: Calculate decay associated difference spectrum from experimental energy scan data and
the convolution of sum of exponential decay and instrumental response function
'''

epilog = '''
*Note
1. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
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



def calc_dads():

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
    parser.add_argument('--escan_file',
                        help='filename for scale corrected energy scan file')
    parser.add_argument('--escan_err_file',
    help='filename for the scaled estimated experimental error of energy scan file')
    parser.add_argument('-t0', '--time_zero', type=float,
                        help='time zero of energy scan')
    parser.add_argument('--escan_time', type=float, nargs='+',
    help='time delay for each energy scan')
    parser.add_argument('--tau', type=float, nargs='+',
                        help='lifetime of each decay path')
    parser.add_argument('--no_base', action='store_false',
    help='Exclude baseline (i.e. very long lifetime component)')
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
    
    if (args.time_zeros is None):
        print('You should set time_zero for energy scan \n')
        return
    else:
        time_zero = args.time_zero

    escan_data = np.genfromtxt(args.escan_file)
    escan_err = np.genfromtxt(args.escan_err_file)
    escan_time = np.array(args.escan_time)
    out_prefix = args.out
    base = args.no_base
    
    ads, ads_eps = dads(escan_time-time_zero, fwhm, tau, base, irf, data=escan_data[:,1:], eps=escan_err)

    e = escan_data[:,0]

    out_ads = np.vstack((e, ads)).T

    # save calculated sads results
    np.savetxt(f'{out_prefix}_dads.txt', out_ads)
    np.savetxt(f'{out_prefix}_dads_eps.txt', ads_eps.T)

    # plot sads results
    plt.title('Decay Associated Difference Spectrum')
    for i in range(ads.shape[0]):
        plt.errorbar(e, ads[i,:], label=f'decay {i+1}')
    plt.legend()
    plt.show()

    return