import argparse
import numpy as np
from ..thy import gen_theory_data


def broadening():
    description = '''
    broadening: generates voigt broadened theoritical calc spectrum
    '''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('peak',
                        help='filename for calculated line shape spectrum')
    parser.add_argument('e_min', type=float,
                        help='minimum energy')
    parser.add_argument('e_max', type=float,
                        help='maximum energy')
    parser.add_argument('e_step', type=float,
    help='energy step')
    parser.add_argument('A', type=float,
                        help='scale factor')
    parser.add_argument('fwhm_G', type=float,
                        help='Full Width at Half Maximum of gaussian shape')
    parser.add_argument('fwhm_L', type=float,
                        help='Full Width at Half Maximum of lorenzian shape')
    parser.add_argument('peak_factor', type=float,
    help='parameter to match descrepency between thoretical spectrum and experimental spectrum')
    parser.add_argument('--scale_energy', action='store_true',
    help='Scaling the energy of peak instead of shifting to match experimental spectrum')
    parser.add_argument('-o', '--out', help='prefix for output files')
    args = parser.parse_args()

    peak = np.genfromtxt(args.peak)
    if args.out is None:
        out = args.prefix
    else:
        out = args.out
    e_min = args.e_min
    e_max = args.e_max
    e_step = args.e_step
    A = args.A
    fwhm_G = args.fwhm_G
    fwhm_L = args.fwhm_L
    peak_factor = args.peak_factor
    if args.scale_energy:
        policy = 'scale'
    else:
        policy = 'shift' 
    e = np.linspace(e_min, e_max, int((e_max-e_min)/e_step)+1)

    broadened_thy = gen_theory_data(e, peak, A, fwhm_G, fwhm_L, peak_factor, policy)

    rescaled_stk = peak
    if policy == 'scale':
        rescaled_stk[:, 0] = rescaled_stk[:, 0]*peak_factor
    else:
        rescaled_stk[:, 0] = rescaled_stk[:, 0] - peak_factor
    rescaled_stk[:, 1] = A*rescaled_stk[:, 1]
    spec_thy = np.vstack((e, broadened_thy))

    np.savetxt(f'{out}_thy_stk.txt', rescaled_stk, fmt=['%.5e', '%.8e'], header='e \t abs')
    np.savetxt(f'{out}_thy.txt', spec_thy.T, fmt=['%.5e', '%.8e'], header='e \t abs')

    return
