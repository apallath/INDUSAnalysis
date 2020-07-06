"""
Integration tests

If run with pytest, verbose outputs are suppressed

Execution times for test cases will be reported to `test_exec_times.txt`
For detailed profiling, run `python -m cProfile test_realdata.py`.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import inspect
import re

from INDUSAnalysis import indus_waters
from INDUSAnalysis.lib import profiling

"""
INDUS waters analysis with sample data
"""

# Waters analysis
@profiling.timefuncfile("test_exec_times.txt")
def test_waters_nopdb():
    if not os.path.exists('waters_test_data'):
        os.makedirs('waters_test_data')

    waters = indus_waters.WatersAnalysis()
    args = ['phiout.dat', 'indus.tpr', 'indus_mol_skip.xtc', '-radius', '6.0', '-skip', '100',
            '-obsstart', '500', '-obspref', 'waters_test_data/obsdata',
            '-window', '50', '-opref', 'waters_test_data/indus',
            '-oformat', 'png', '-dpi', '150',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    waters.parse_args(args)
    waters.read_args()
    waters()


# Waters analysis
@profiling.timefuncfile("test_exec_times.txt")
def test_waters_pdb():
    if not os.path.exists('waters_test_data'):
        os.makedirs('waters_test_data')

    waters = indus_waters.WatersAnalysis()
    args = ['phiout.dat', 'indus.tpr', 'indus_mol_skip.xtc', '-radius', '6.0', '-skip', '100',
            '-obsstart', '500', '-obspref', 'waters_test_data/obsdata',
            '-window', '50', '-opref', 'waters_test_data/indus',
            '-oformat', 'png', '-dpi', '150',
            '--genpdb', '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    waters.parse_args(args)
    waters.read_args()
    waters()


# Waters analysis, replotting from saved data, with PDB generation
@profiling.timefuncfile("test_exec_times.txt")
def test_waters_replot_pdb():
    if not os.path.exists('waters_test_data'):
        os.makedirs('waters_test_data')

    waters = indus_waters.WatersAnalysis()
    args = ['phiout.dat', 'indus.tpr', 'indus_mol_skip.xtc', '-radius', '6.0', '-skip', '100',
            '--replot', '-replotpref', 'waters_test_data/indus',
            '-obsstart', '500', '-obspref', 'waters_test_data/obsdata',
            '-window', '50', '-opref', 'waters_test_data/indus',
            '-oformat', 'png', '-dpi', '150',
            '--genpdb', '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    waters.parse_args(args)
    waters.read_args()
    waters()


"""
Protein order parameters analysis with actual data
@timefuncfile("test_exec_times.txt")
def test_order_params_nopdb():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    op = OrderParams()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'order_params_test_data/indus', '-oformat', 'png',
                   '-dpi', '150', '-align', 'backbone', '-select', 'backbone', '-window', '2000',
                   '-skip', '100',
                   '--remote']
    if __name__=="__main__":
        args.append("--verbose")
    op.parse_args(args)
    op.read_args()
    op()
    return True

@timefuncfile("test_exec_times.txt")
def test_order_params_pdb():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    op = OrderParams()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'order_params_test_data/indus', '-oformat', 'png',
                   '-dpi', '150', '-align', 'backbone', '-select', 'backbone', '-window', '2000',
                   '-skip', '100',
                   '--genpdb', '--remote']
    if __name__=="__main__":
        args.append("--verbose")
    op.parse_args(args)
    op.read_args()
    op()
    return True

@timefuncfile("test_exec_times.txt")
def test_order_params_replot_pdb():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    op = OrderParams()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'order_params_test_data/indus', '-oformat', 'png',
                   '--replot', '-replotpref', 'order_params_test_data/indus',
                   '-dpi', '150', '-align', 'backbone', '-select', 'backbone', '-window', '2000',
                   '-skip', '100',
                   '--genpdb', '--remote']
    if __name__=="__main__":
        args.append("--verbose")
    op.parse_args(args)
    op.read_args()
    op()
    return True

# 3res-sh method, with no PDB generation
@timefuncfile("test_exec_times.txt")
def test_contacts_3res_sh_nopdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = Contacts()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'contacts_test_data/indus', '-oformat', 'png',
                    '-method', '3res-sh',
                    '-dpi', '150', '-distcutoff', '4.5', '-skip', '100', '-bins', '50', '--remote']
    if __name__=="__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()

# 3res-sh method, with PDB generation
@timefuncfile("test_exec_times.txt")
def test_contacts_3res_sh_pdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = Contacts()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'contacts_test_data/indus', '-oformat', 'png',
                    '-method', '3res-sh',
                    '-dpi', '150', '-distcutoff', '4.5', '-skip', '100', '-bins', '50', '--remote',
                    '--genpdb']
    if __name__=="__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()

# 3res-sh method, replotting from saved data, with PDB generation
@timefuncfile("test_exec_times.txt")
def test_contacts_3res_sh_replot_pdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = Contacts()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'contacts_test_data/indus', '-oformat', 'png',
                    '-method', '3res-sh', '--replot', '-replotpref', 'contacts_test_data/indus',
                    '-dpi', '150', '-distcutoff', '4.5', '-skip', '100', '-bins', '50', '--remote',
                    '--genpdb']
    if __name__=="__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()

    return True

# atomic-sh method, with no PDB generation
@timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_nopdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = Contacts()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'contacts_test_data/indus', '-oformat', 'png',
                    '-method', 'atomic-sh',
                    '-dpi', '150', '-distcutoff', '5', '-skip', '100', '-bins', '50', '--remote']
    if __name__=="__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()

# atomic-sh method, with PDB generation
@timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_pdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = Contacts()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'contacts_test_data/indus', '-oformat', 'png',
                    '-method', 'atomic-sh',
                    '-dpi', '150', '-distcutoff', '5', '-skip', '100', '-bins', '50', '--remote',
                    '--genpdb']
    if __name__=="__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()

# atomic-sh method, replotting from saved data, with PDB generation
@timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_replot_pdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = Contacts()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-opref', 'contacts_test_data/indus', '-oformat', 'png',
                    '-method', 'atomic-sh', '--replot', '-replotpref', 'contacts_test_data/indus',
                    '-dpi', '150', '-distcutoff', '5', '-skip', '100', '-bins', '50', '--remote',
                    '--genpdb']
    if __name__=="__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()

    return True
"""


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
