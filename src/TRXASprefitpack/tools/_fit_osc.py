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
from ..driver import print_DriverResult, plot_DriverResult, save_DriverResult
from ..driver import fit_transient_dmp_osc
from .misc import read_data

description = '''
fit osc: fitting residual of experimental time trace spectrum data with the convolution of the sum of damped oscilliation and irf function
There are three types of irf function (gaussian, cauchy, pseudo voigt)
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

method_glb_help='''
Global optimization Method
 'ampgo': Adapted Memory Programming for Global Optimization Algorithm
 'basinhopping'
'''

def fit_osc():

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
    parser.add_argument('prefix', nargs='+',
                        help='prefix of residual of time delay scan data to fit' +
                        'It will prefix_i.txt')
    parser.add_argument('--num_file', type=int, nargs='+',
                         help='number of scan file corresponding to each prefix')
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
    parser.add_argument('--method_glb', default='ampgo', choices=['ampgo', 'basinhopping'],
    help=method_glb_help)
    parser.add_argument('-o', '--outdir', default='out',
                        help='directory for output files')
    args = parser.parse_args()

    irf = args.irf
    if irf == 'g':
        if args.fwhm_G is None:
            print('You are using gaussian irf, so you should set fwhm_G!\n')
            return
        else:
            fwhm_init = args.fwhm_G
    elif irf == 'c':
        if args.fwhm_L is None:
            print('You are using cauchy/lorenzian irf,' +
                  'so you should set fwhm_L!\n')
            return
        else:
            fwhm_init = args.fwhm_L
    else:
        if (args.fwhm_G is None) or (args.fwhm_L is None):
            print('You are using pseudo voigt irf,' +
                  'so you should set both fwhm_G and fwhm_L!\n')
            return
        else:
            fwhm_init = np.array([args.fwhm_G, args.fwhm_L])

    if (args.time_zeros is None) and (args.time_zeros_file is None):
        print('You should set either time_zeros or time_zeros_file!\n')
        return
    elif args.time_zeros is None:
        t0_init = np.genfromtxt(args.time_zeros_file)
    else:
        t0_init = np.array(args.time_zeros)

    prefix = np.array(args.prefix, dtype=str)
    num_file = np.array(args.num_file, dtype=int)

    t = np.empty(prefix.size, dtype=object)
    data = np.empty(prefix.size, dtype=object)
    eps = np.empty(prefix.size, dtype=object)
    num_scan = np.sum(num_file)

    for i in range(prefix.size):
        t[i] = np.genfromtxt(f'{prefix[i]}_1.txt')[:, 0]
        num_data_pts = t[i].size
        data[i], eps[i] = read_data(prefix[i], num_file[i], num_data_pts, 10)

    print(f'fitting with total {num_scan} data set!\n')

    tau_init = np.array(args.tau)
    period_init = np.array(args.period)
    phase_init = np.array(args.phase)

    bound_fwhm = None; bound_t0 = None
    if args.fix_irf:
        if irf in ['g', 'c']:
            bound_fwhm = [(fwhm_init, fwhm_init)]
        else:
            bound_fwhm = [(fwhm_init[0], fwhm_init[0]), (fwhm_init[1], fwhm_init[1])]
    if args.fix_t0:
        bound_t0 = t0_init.size*[None]
        for i in range(t0_init.size):
            bound_t0[i] = (t0_init[i], t0_init[i])
    
    result = fit_transient_dmp_osc(irf, fwhm_init, t0_init, tau_init, period_init, phase_init,
    method_glb=args.method_glb, bound_fwhm=bound_fwhm, bound_t0=bound_t0, t=t, data=data, eps=eps)

    print(print_DriverResult(result, prefix))
    plot_DriverResult(result, prefix, t=t, data=data, eps=eps)
    save_DriverResult(result, args.outdir, prefix, t, eps)

    return
