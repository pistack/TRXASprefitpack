# fit eq py
# Wrapper script for fit_eq()
# Date: 2022. 6. 21.
# Author: pistack
# Email: pistatex@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import fit_eq

if __name__ == "__main__":
    fit_eq()
