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
    parser.add_argument('A', type=float,
                        help='scale factor')
    parser.add_argument('fwhm_G', type=float,
                        help='Full Width at Half Maximum of gaussian shape')
    parser.add_argument('fwhm_L', type=float,
                        help='Full Width at Half Maximum of lorenzian shape')
    parser.add_argument('peak_shift', type=float,
                        help='discrepancy of peak position between theory ' +
                        'and experiment')
    parser.add_argument('-o', '--out', default='out',
                        help='prefix for output files')
    args = parser.parse_args()

    peak = np.genfromtxt(args.peak)
    out = args.out
    e_min = args.e_min
    e_max = args.e_max
    A = args.A
    fwhm_G = args.fwhm_G
    fwhm_L = args.fwhm_L
    peak_shift = args.peak_shift
    e = np.linspace(e_min, e_max, int((e_max-e_min)*100)+1)

    gen_theory_data(e, peak, A, fwhm_G, fwhm_L, peak_shift,
                    out=out)

    return
