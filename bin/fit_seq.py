# fit seq py
# Wrapper script for fit_seq()
# Date: 2022. 6. 12.
# Author: pistack
# Email: pistatex@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import fit_seq

if __name__ == "__main__":
    fit_seq()
