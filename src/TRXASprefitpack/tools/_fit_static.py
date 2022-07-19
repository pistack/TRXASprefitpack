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
from ..driver import print_StaticResult, save_StaticResult
from ..driver import fit_static_voigt
from .misc import plot_StaticResult

description = '''
fit static: fitting static spectrum with theoretically calculated line spectrum
broadened by spectral line shape
v: voigt profile,
g: gaussian,
l: lorenzian,
Moreover, it uses linear baseline to correct baseline feature of experimental spectrum
'''

epilog = ''' 
*Note
 If fwhm_G of voigt component is zero then this voigt component is treated as lorenzian
 If fwhm_L of voigt component is zero then this voigt component is treated as gaussian
'''

method_glb_help='''
Global optimization Method
 'ampgo': Adapted Memory Programming for Global Optimization Algorithm
 'basinhopping'
'''

edge_help = '''
Type of edge function if not set, edge is not included.
 'g': gaussian type edge function
 'l': lorenzian type edge function
'''


def fit_static():

    tmp = argparse.RawDescriptionHelpFormatter
    parse = argparse.ArgumentParser(formatter_class=tmp,
                                    description=description,
                                    epilog=epilog)
    parse.add_argument('filename', help='filename for experimental spectrum')
    parse.add_argument('--e0_voigt', type=float, nargs='*',
    help='peak position of each voigt component')
    parse.add_argument('--fwhm_G_voigt', type=float, nargs='*',
                        help='full width at half maximum for gaussian shape ' +
                        'It would be not used when you set lorenzian line shape')
    parse.add_argument('--fwhm_L_voigt', type=float, nargs='*',
                        help='full width at half maximum for lorenzian shape ' +
                        'It would be not used when you use gaussian line shape')
    parse.add_argument('--edge', type=str, choices=['g', 'l'],
    help=edge_help)
    parse.add_argument('--e0_edge', type=float,
    help='edge position')
    parse.add_argument('--fwhm_edge', type=float,
    help='full width at half maximum parameter of edge')
    parse.add_argument('--base_order', type=int,
    help ='Order of polynomial to correct baseline feature. If it is not set then baseline is not corrected')
    parse.add_argument('-o', '--outdir', default='out', help='directory to store output file')
    parse.add_argument('--method_glb', default='basinhopping', choices=['ampgo', 'basinhopping'],
    help=method_glb_help)

    args = parse.parse_args()

    filename = args.filename
    if args.e0_voigt is None:
        e0_init = None
        fwhm_G_init = None
        fwhm_L_init = None
    else:
        e0_init = np.array(args.e0_voigt)
        fwhm_G_init = np.array(args.fwhm_G_voigt)
        fwhm_L_init = np.array(args.fwhm_L_voigt)
    
    edge = args.edge
    e0_edge_init = args.e0_edge
    fwhm_edge_init = args.fwhm_edge
    base_order = args.base_order
    method_glb = args.method_glb
    outdir = args.outdir

    tmp = np.genfromtxt(filename)
    e = tmp[:, 0]
    data = tmp[:, 1]
    if tmp.shape[1] == 2:
        eps = np.max(np.abs(data))/1000*np.ones_like(e)
    else:
        eps = tmp[:, 2]

    result = fit_static_voigt(e0_init, fwhm_G_init, fwhm_L_init, edge, e0_edge_init, fwhm_edge_init,
    base_order, method_glb, e=e, data=data, eps=eps)

    print(print_StaticResult(result))
    plot_StaticResult(result, e=e, data=data, eps=eps)
    save_StaticResult(result, outdir, e=e, eps=eps)

    return
