# pylint: disable = missing-module-docstring, wrong-import-position
# fit tscan gui py
# Wrapper script for fit_tscan_gui()
# Date: 2022. 12. 20.
# Author: pistack
# Email: pistack@yonsei.ac.kr

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")
from TRXASprefitpack.tools._fit_tscan_gui import fit_tscan_gui

if __name__ == '__main__':
    fit_tscan_gui()
