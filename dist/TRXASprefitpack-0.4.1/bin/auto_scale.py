# broadening py
# Wrapper script for auto_scale()
# Date: 2021. 9. 1.
# Author: pistack
# Email: pistatex@yonsei.ac.kr

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.tools import auto_scale

if __name__ == "__main__":
    auto_scale()
