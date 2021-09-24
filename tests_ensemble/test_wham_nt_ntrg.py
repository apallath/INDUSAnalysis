"""
Integration tests for wham_nt_ntrg

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
def test_ensemble_polymers_wham_nt_ntrg_C45():
    if not os.path.exists('wham_nt_ntrg_test_data'):
        os.makedirs('wham_nt_ntrg_test_data')
    anl = WHAM_analysis_biasN()

    # 1D-tests

    anl.get_test_data()
    anl.get_test_data2()
    anl.plot_hist()
    anl.run_binless_log_likelihood()
    anl.run_kappa_checks()
    anl.run_reweighting_checks()
    anl.run_phi_ensemble_reweight()
    anl.run_reweight_phi_1_star()
    anl.find_basins()
    anl.run_phi_e_star_opt()
    anl.calc_deltaGu_diff_method()
    anl.run_phi_c_star_opt()
    anl.calc_deltaGu_int_method_1D()

    # 1D-boot test

    anl.run_bootstrap_ll_phi_ensemble()

    # 2D-tests
    anl.run_2D_binless_log_likelihood()
    anl.run_2D_bin_Rg()
    anl.run_2D_reweight_phi_star()
    anl.run_2D_reweight_phi_star_bin_Rg()
    anl.run_coex_integration_2D()
    anl.run_coex_integration_Rg()


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print(obj[0])
            obj[1]()
