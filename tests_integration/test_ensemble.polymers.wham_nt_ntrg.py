"""
Integration tests for wham_nt_ntrg

Execution times for test cases will be reported to `test_exec_times.txt`
For detailed profiling, run `python -m cProfile test_realdata.py`.
"""

import os
import sys
import inspect
import re

from INDUSAnalysis.lib import profiling
from INDUSAnalysis.ensemble.polymers.wham_nt_ntrg import WHAM_analysis_biasN


@profiling.timefuncfile("test_exec_times.txt")
def test_ensemble_polymers_wham_nt_ntrg_C45():
    if not os.path.exists('wham_nt_ntrg_test_data'):
        os.makedirs('wham_nt_ntrg_test_data')
    anl = WHAM_analysis_biasN()
    anl.get_test_data()
    anl.get_test_data2()


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
