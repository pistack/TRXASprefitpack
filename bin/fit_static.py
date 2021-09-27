# fit static py
# Wrapper script for fit_static()
# Date: 2021. 8. 30.
# Author: pistack
# Email: pistatex@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import fit_static

if __name__ == "__main__":
    fit_static()

