# calc_sads.py
# Wrapper script for calc_sads()
# Date: 2022. 7. 12.
# Author: pistack
# Email: pistack@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import calc_sads

if __name__ == "__main__":
    calc_sads()
