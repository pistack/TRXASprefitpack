# pylint: disable = missing-module-docstring, wrong-import-position
# calc dads gui qt py
# Wrapper script for fit_tscan_gui()
# Date: 2026. 09. 04.
# Author: pistack
# Email: phistack@kaist.ac.kr

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")
from TRXASprefitpack.gui.app_calc_dads import main

if __name__ == '__main__':
    main()
