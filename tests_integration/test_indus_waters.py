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

import numpy as np

from INDUSAnalysis.timeseries import TimeSeriesAnalysis
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

# Make sure that INDUSAnalysis works when reading waters from a corrupt file
@profiling.timefuncfile("test_exec_times.txt")
def test_waters_nopdb_corrupt_check():
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

    waters_corrupt = indus_waters.WatersAnalysis()
    args_corrupt = ['phiout_corrupt.dat', 'indus.tpr', 'indus_mol_skip.xtc', '-radius', '6.0', '-skip', '100',
            '-obsstart', '500', '-obspref', 'waters_test_data/obsdata',
            '-window', '50',
            '-opref', 'waters_test_data/indus_corrupt', '-oformat', 'png', '-dpi', '150',
            '--remote']
    if __name__ == "__main__":
        args_corrupt.append("--verbose")
    waters_corrupt.parse_args(args_corrupt)
    waters_corrupt.read_args()
    waters_corrupt()

    # Check equality
    ntw = TimeSeriesAnalysis.load_TimeSeries("waters_test_data/indus_Ntw.pkl")
    corrupt_ntw = TimeSeriesAnalysis.load_TimeSeries("waters_test_data/indus_corrupt_Ntw.pkl")
    assert(np.allclose(ntw.data_array, corrupt_ntw.data_array))
    assert(np.allclose(ntw.time_array, corrupt_ntw.time_array))

    n = TimeSeriesAnalysis.load_TimeSeries("waters_test_data/indus_N.pkl")
    corrupt_n = TimeSeriesAnalysis.load_TimeSeries("waters_test_data/indus_corrupt_N.pkl")
    assert(np.allclose(n.data_array, corrupt_n.data_array))
    assert(np.allclose(n.time_array, corrupt_n.time_array))

    ni = TimeSeriesAnalysis.load_TimeSeries("waters_test_data/indus_probe_waters.pkl")
    corrupt_ni = TimeSeriesAnalysis.load_TimeSeries("waters_test_data/indus_corrupt_probe_waters.pkl")
    assert(np.allclose(ni.data_array, corrupt_ni.data_array))
    assert(np.allclose(ni.time_array, corrupt_ni.time_array))

if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
