# pylint: disable = missing-module-docstring, wrong-import-position
# calc dads gui py
# Wrapper script for fit_tscan_gui()
# Date: 2022. 12. 26.
# Author: pistack
# Email: pistack@yonsei.ac.kr

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")
from TRXASprefitpack.tools._calc_dads_gui import calc_dads_gui

if __name__ == '__main__':
    calc_dads_gui()
