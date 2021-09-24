"""
Integration tests for critical_cluster

Execution times for test cases will be reported to `test_exec_times.txt`
For detailed profiling, run `python -m cProfile test_realdata.py`.
"""

import os
import sys
import inspect
import logging
import re

from INDUSAnalysis.lib import profiling
from INDUSAnalysis.ensemble.polymers.wham_nt_ntrg import WHAM_analysis_biasN

# Logging
logging.basicConfig(level=logging.DEBUG)


@profiling.timefuncfile("test_exec_times.txt")
def test_ensemble_polymers_critical_cluster():
    if not os.path.exists('critical_cluster_test_data'):
        os.makedirs('critical_cluster_test_data')


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
