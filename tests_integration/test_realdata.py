"""
Integration tests

For detailed profiling, run `python -m cProfile test_realdata.py`.
For profiling only each test, run `python test_realdata.py` instead of pytest,
and the timefunc decorators will handle run time analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from analysis.indus_waters import IndusWaters
from analysis.protein_order_params import OrderParams
from analysis.contacts import Contacts
from meta_analysis.profiling import timefunc

"""Ensure that INDUS waters analysis does not break on running with actual data"""
@timefunc
def test_waters():
    if not os.path.exists('waters_test_data'):
        os.makedirs('waters_test_data')

    forward = IndusWaters()
    forward.parse_args(['phiout_fwd.dat', '-obsstart', '500', '-obspref',
                       'waters_test_data/obsdata', '-window', '50', '-opref', 'waters_test_data/fwd', '-oformat',
                       'png', '-dpi', '150', '--remote'])
    forward.read_args()
    forward()

    back = IndusWaters()
    back.parse_args(['phiout_back.dat', '-obsstart', '500', '-obspref',
                     'waters_test_data/obsdata', '-window', '50', '-opref', 'waters_test_data/back', '-oformat',
                     'png', '-dpi', '150', '-apref', 'waters_test_data/fwd', '-aprevlegend',
                     'Biased with INDUS', '-acurlegend', 'Unbiased',
                     '--remote'])
    back.read_args()
    back()
    return True

"""Ensure that protein order parameters analysis does not break on running
with actual data"""
@timefunc
def test_order_params():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    forward = OrderParams()
    forward.parse_args(['prod.gro', 'indus_fwd_mol.xtc', '-opref', 'order_params_test_data/fwd', '-oformat', 'png',
                        '-dpi', '150', '-align', 'backbone', '-select', 'backbone', '-window', '50',
                        '--remote'])
    forward.read_args()
    forward()

    back = OrderParams()
    back.parse_args(['prod.gro', 'indus_back_mol.xtc', '-reftrajf', 'indus_fwd_mol.xtc',
                     '-reftstep', '0', '-opref', 'order_params_test_data/back', '-apref', 'order_params_test_data/fwd',
                     '-aprevlegend', "Biased", '-acurlegend', "Biasing off", '-oformat', 'png',
                     '-dpi', '150', '-align', 'backbone', '-select', 'backbone', '-window', '50',
                     '--remote'])
    back.read_args()
    back()
    return True

"""Ensure that contacts analysis does not break on running
with actual data"""
@timefunc
def test_contacts():
    return True

if __name__=="__main__":
    test_waters()
    test_order_params()
    test_contacts()
