import numpy as np
import matplotlib.pyplot as plt
import os

from analysis.indus_waters import IndusWaters

"""Ensure that INDUS waters analysis does not break on running with actual data"""

def test_realdata():
    if not os.path.exists('waters_test_data'):
        os.makedirs('waters_test_data')
        
    forward = IndusWaters()
    forward.parse_args(['phiout-4.25.dat', '-obsstart', '500', '-obspref',
                       'waters_test_data/obsdata', '-window', '50', '-opref', 'waters_test_data/fwd', '-oformat',
                       'png', '-dpi', '150', '--remote'])
    forward.read_args()
    forward()

    back = IndusWaters()
    back.parse_args(['phiout0.0.dat', '-obsstart', '500', '-obspref',
                       'waters_test_data/obsdata', '-window', '50', '-opref', 'waters_test_data/back', '-oformat',
                       'png', '-dpi', '150', '-apref', 'waters_test_data/fwd', '-aprevlegend',
                       'Biased with INDUS', '-acurlegend', 'Unbiased',
                       '--remote'])
    back.read_args()
    back()
    return True
