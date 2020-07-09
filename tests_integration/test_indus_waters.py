"""
Integration tests for indus_waters.

If run with pytest, verbose outputs are suppressed.

Execution times for test cases will be reported to `test_exec_times.txt`
For detailed profiling, run `python -m cProfile test_realdata.py`.
"""

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
            '-window', '50',
            '-opref', 'waters_test_data/indus', '-oformat', 'png', '-dpi', '150',
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
            '-window', '50',
            '-opref', 'waters_test_data/indus', '-oformat', 'png', '-dpi', '150',
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
            '-window', '50',
            '-opref', 'waters_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--genpdb', '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    waters.parse_args(args)
    waters.read_args()
    waters()


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
