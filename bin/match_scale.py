# match_scale.py
# Wrapper script for match_scale()
# Date: 2022. 7. 11.
# Author: pistack
# Email: pistack@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import match_scale

if __name__ == "__main__":
    match_scale()
