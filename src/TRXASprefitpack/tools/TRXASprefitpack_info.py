import argparse
from ..doc import __info__


def TRXASprefitpack_info():
    parser = argparse.ArgumentParser(description='Give information about '+
                                     'the function defined in ' +
                                     'TRXASprefitpack')
    parser.add_argument('-fn', '--fun_name',
                        help="function name")
    arg = parser.parse_args()
    if arg.fun_name in __info__.keys():
        print(__info__[arg.fun_name])
    else:
        print(__info__['description'])
        print(__info__['licence'])
        print("Type TRXASprefitpack lgpl-3.0 to see full description of the licence")
    return
        
