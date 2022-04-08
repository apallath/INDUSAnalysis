"""
Integration tests for contacts.

If run with pytest, verbose outputs are suppressed.

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
def test_contacts_alk_ua_r4_5_n0():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['c30.tpr', 'c30.xtc',
            '-method', 'alk-ua', '-distcutoff', '4.5', '-connthreshold', '0', '-skip', '1', '-bins', '10',
            '-obsstart', '0',
            '-opref', 'contacts_test_data/poly_r4.5_n0', '-oformat', 'png', '-dpi', '300',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_alk_ua_r4_5_n0_poly():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['c30_poly.gro', 'c30_poly.xtc',
            '-apsp_structf', 'c30.tpr',
            '-method', 'alk-ua', '-distcutoff', '4.5', '-connthreshold', '0', '-skip', '1', '-bins', '10',
            '-obsstart', '0',
            '-opref', 'contacts_test_data/poly_r4.5_n0', '-oformat', 'png', '-dpi', '300',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_h_r7_n0():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-h', '-distcutoff', '7.0', '-connthreshold', '0', '-skip', '50', '-bins', '10',
            '-obsstart', '500',
            '-opref', 'contacts_test_data/indus_r7_n0', '-oformat', 'png', '-dpi', '300',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_h_r7_n5():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-h', '-distcutoff', '7.0', '-connthreshold', '5', '-skip', '50', '-bins', '10',
            '-obsstart', '500',
            '-opref', 'contacts_test_data/indus_r7_n5', '-oformat', 'png', '-dpi', '300',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_r7_n0():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-sh', '-distcutoff', '7.0', '-connthreshold', '0', '-skip', '50', '-bins', '10',
            '-obsstart', '500',
            '-opref', 'contacts_test_data/indus_r7_n0', '-oformat', 'png', '-dpi', '300',
            '--remote']
    if __name__ == "__main__":
        args.append("--verbose")
    cts.parse_args(args)
    cts.read_args()
    cts()


@profiling.timefuncfile("test_exec_times.txt")
def test_contacts_atomic_sh_r7_n5():
    if not os.path.exists('contacts_test_data'):
        os.makedirs('contacts_test_data')

    cts = contacts.ContactsAnalysis()
    args = ['indus.tpr', 'indus_mol_skip.xtc',
            '-method', 'atomic-sh', '-distcutoff', '7.0', '-connthreshold', '5', '-skip', '50', '-bins', '10',
            '-obsstart', '500',
            '-opref', 'contacts_test_data/indus_r7_n5', '-oformat', 'png', '-dpi', '300',
            '--remote']
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
