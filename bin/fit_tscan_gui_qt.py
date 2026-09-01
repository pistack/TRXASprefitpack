# pylint: disable = missing-module-docstring, wrong-import-position
# fit tscan gui py
# Wrapper script for qt version of fit_tscan_gui()
# Date: 2026. 09. 01.
# Author: pistack
# Email: phistack@kaist.ac.kr

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")
from TRXASprefitpack.gui.app_fit_tscan import main

if __name__ == '__main__':
    main()
