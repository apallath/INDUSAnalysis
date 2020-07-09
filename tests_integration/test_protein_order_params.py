"""
Integration tests for protein_order_params

If run with pytest, verbose outputs are suppressed

Execution times for test cases will be reported to `test_exec_times.txt`
For detailed profiling, run `python -m cProfile test_realdata.py`.
"""

import os
import sys
import inspect
import re

from INDUSAnalysis import protein_order_params
from INDUSAnalysis.lib import profiling

"""
Protein order parameters analysis with actual data
"""
@profiling.timefuncfile("test_exec_times.txt")
def test_order_params_nopdb():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    op = protein_order_params.OrderParamsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-align', 'backbone', '-select', 'backbone', '-skip', '100',
            '-window', '2',
            '-opref', 'order_params_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--remote']

    if __name__ == "__main__":
        args.append("--verbose")
    op.parse_args(args)
    op.read_args()
    op()
    return True


@profiling.timefuncfile("test_exec_times.txt")
def test_order_params_pdb():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    op = protein_order_params.OrderParamsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-align', 'backbone', '-select', 'backbone', '-skip', '100',
            '-window', '2',
            '-opref', 'order_params_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--genpdb', '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    op.parse_args(args)
    op.read_args()
    op()
    return True


@profiling.timefuncfile("test_exec_times.txt")
def test_order_params_replot_pdb():
    if not os.path.exists('order_params_test_data'):
        os.makedirs('order_params_test_data')

    op = protein_order_params.OrderParamsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc', '-align', 'backbone', '-select', 'backbone', '-skip', '100',
            '--replot', '-replotpref', 'order_params_test_data/indus',
            '-window', '2',
            '-opref', 'order_params_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--genpdb', '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    op.parse_args(args)
    op.read_args()
    op()
    return True


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
