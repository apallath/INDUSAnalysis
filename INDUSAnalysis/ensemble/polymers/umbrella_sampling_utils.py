"""
Functions to help perform umbrella sampling calculations.

This module can also be called as a stand-alone script, given command line options.
"""
import argparse

from INDUSAnalysis import timeseries
from INDUSAnalysis.indus_waters import WatersAnalysis

def estimate_kappa(datf: str, temp: float, start_time: float = 0, end_time: float = None):
    """
    Estimates kappa based on var(N~) from an unbiased INDUS simulation.

    beta * kappa = 2 / var(Nt) to 5 / var(Nt).

    Args:
        datf (str): Path to INDUS waters data file.
        temp (float): Simulation temperature (in K).
        start_time (float): Time to begin computation of variance at.
        end_time (float): Time to end computation of variance at.

    Returns:
        kappa_range (tuple): Estimated range of kappa in kJ/mol (kappa_lower: float, kappa_upper: float)
    """
    ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(datf)
    var_Ntw = ts_Ntw[start_time:end_time].std() ** 2
    return (8.314 / 1000 * temp * 2 / var_Ntw, 8.314 / 1000 * temp * 5 / var_Ntw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("calc_type", help="Type of calculation (options = est_kappa)")

    est_kappa_args = parser.add_argument_group("kappa estimation arguments")
    est_kappa_args.add_argument("-watersf", help="INDUS waters data file")
    est_kappa_args.add_argument("-temp", type=float, help="Simulation temperature")
    est_kappa_args.add_argument("-tstart", type=float, help="Time to begin computation of variance at", default=0)
    est_kappa_args.add_argument("-tend", type=float, help="Time to end computation of variance at", default=None)

    args = parser.parse_args()

    if args.calc_type == "est_kappa":
        print(estimate_kappa(args.watersf, args.temp, args.tstart, args.tend))
