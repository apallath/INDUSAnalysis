"""
Integration tests for contacts

If run with pytest, verbose outputs are suppressed

Execution times for test cases will be reported to `test_exec_times.txt`
For detailed profiling, run `python -m cProfile test_realdata.py`.
"""

import os
import sys
import inspect
import re

from INDUSAnalysis import contacts
from INDUSAnalysis.lib import profiling


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_nopdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-sh', '-distcutoff', '5', '-skip', '100', '-bins', '50',
            '-opref', 'contacts_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_pdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-sh', '-distcutoff', '5', '-skip', '100', '-bins', '50',
            '-opref', 'contacts_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--remote', '--genpdb']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_replot_pdb():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-sh', '-distcutoff', '5', '-skip', '100', '-bins', '50',
            '-opref', 'contacts_test_data/indus', '-oformat', 'png', '-dpi', '150',
            '--replot', '-replotpref', 'contacts_test_data/indus',
            '--remote', '--genpdb']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
