# TRXASprefitpack_info py
# Wrapper script for TRXASprefitpack_info()
# Date: 2021. 8. 30.
# Author: pistack
# Email: pistatex@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import TRXASprefitpack_info

if __name__ == "__main__":
    TRXASprefitpack_info()
