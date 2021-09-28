"""
Calculates critical cluster size for solvated polymers.
"""
import argparse
from collections import OrderedDict
import logging
from multiprocessing import Pool
import os
import pickle
import yaml

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.optimize
from scipy.signal import find_peaks
from scipy.special import logsumexp
from WHAM.lib import potentials, timeseries
import WHAM.binless
import WHAM.statistics

from INDUSAnalysis.indus_waters import WatersAnalysis
from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis


# Logging
logger = logging.getLogger(__name__)

# Disable matplotlib logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.ERROR)


class CriticalClusterAnalysis:
    """
    Class for performing calculations and plotting figures.

    All configuration data is read from and written to a YAML configuration file.

    Args:
        config_file: Path to config yaml file (default="config.yaml")

    Attributes:
        config: Dictionary containing all configuration parameters
    """
    def __init__(self, config_file="cluster.yaml"):
        self.config_file = config_file
        self.register()

    def register(self):
        self.func_registry = OrderedDict([
            ("get", self.get_data),
            ("get_full", self.get_full_waters_data),
            ("ni_avg", self.compute_boxes_ni_averages),
            ("ni_cut", self.compute_boxes_ni_cutoffs),
            ("ni_bin", self.compute_boxes_ni_binary),
            ("sample_ni", self.sample_boxes_ni_binary),
            ("ci_avg", self.compute_boxes_ci_averages),
            ("ci_cut", self.compute_boxes_ci_cutoffs),
            ("ci_bin", self.compute_boxes_ci_binary),
            ("sample_ci", self.sample_boxes_ci_binary),
            ("ni_ci_bin", self.compute_boxes_ni_ci_binary),
            ("sample_ni_ci", self.sample_boxes_ni_ci_binary)
        ])
        return self.func_registry

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def update_config(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    ############################################################################
    # data loaders
    ############################################################################

    def get_data(self):
        """Gets all window data from original umbrella sampling simulations (without solvent coordinates)."""
        # Load config
        self.load_config()

        # N* associated with each window
        # First window is unbiased
        n_star_win = ["unbiased"]

        biased_windows = list(self.config["windows"].keys())
        biased_windows.remove("unbiased")

        n_star_win.extend(sorted(biased_windows))

        # Raw, correlated timeseries CV data from each window
        Ntw_win = []

        # Read waters
        for n_star in n_star_win:
            ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(self.config["windows"][n_star]["Nt_file"])
            NTSCALE = int(self.config["windows"][n_star]["XTCDT"] / self.config["windows"][n_star]["UMBDT"])
            Ntw_win.append(ts_Ntw[self.config["data_collection"]["tstart"]:self.config["data_collection"]["tend"]:NTSCALE * self.config["data_collection"]["base_samp_freq"]].data_array)
            logger.debug("(N~) N*={}: {} to end, skipping {}. {} entries.".format(n_star, self.config["data_collection"]["tstart"], self.config["data_collection"]["base_samp_freq"],
                         len(ts_N[self.config["data_collection"]["tstart"]:self.config["data_collection"]["tend"]:NTSCALE * self.config["data_collection"]["base_samp_freq"]].data_array)))

        tsa = TimeSeriesAnalysis()

        Rg_win = []

        for n_star in n_star_win:
            ts = tsa.load_TimeSeries(self.config["windows"][n_star]["Rg_file"])
            Rg_win.append(ts[self.config["data_collection"]["tstart"]:self.config["data_collection"]["tend"]:self.config["data_collection"]["base_samp_freq"]].data_array)
            logger.debug("(Rg) N*={}: {} to end, skipping {}. {} entries.".format(n_star, self.config["data_collection"]["tstart"], self.config["data_collection"]["base_samp_freq"],
                         len(ts[self.config["data_collection"]["tstart"]:self.config["data_collection"]["tend"]:self.config["data_collection"]["base_samp_freq"]].data_array)))

        return n_star_win, Ntw_win, Rg_win

    def get_full_waters_data(self):
        """Gets all window data from full umbrella sampling simulations (with solvent coordinates)."""
        # Load config
        self.load_config()

        raise NotImplementedError()

    ############################################################################
    # boxes and windows identification
    ############################################################################

    def plot_boxes(self):
        """Plots pre-computed WHAM free energy profile at phi* and superimposes C, IC, IE and E boxes on top of it."""
        # Load config
        self.load_config()

    def get_windows_in_boxes(self):
        """Computes how many data points from each window fall into each box."""
        # Load config
        self.load_config()

    ############################################################################
    # solvation & binary solvation calc
    ############################################################################

    def compute_boxes_ni_averages(self):
        """Computes average ni within each box."""
        # Load config
        self.load_config()

    def compute_boxes_ni_cutoffs(self):
        """Computes desolvation ni cutoffs as (<ni>C + <ni>E) / 2."""
        # Load config
        self.load_config()

    def compute_boxes_ni_binary(self):
        """For each simulation snapshot in the box, computes the binary solvation state of atoms."""
        # Load config
        self.load_config()

    def sample_boxes_ni_binary(self):
        """Draws samples from boxes based on the biased probability in the phi* ensemble,
        and generates a PDB with atoms colored by their solvation state."""
        # Load config
        self.load_config()

    ############################################################################
    # contact density & binary contacts calc
    ############################################################################

    def compute_boxes_ci_averages(self):
        """Computes average ci within each box."""
        # Load config
        self.load_config()

    def compute_boxes_ci_cutoffs(self):
        """Computes contact density ci cutoffs as (<ci>C + <ci>E) / 2."""
        # Load config
        self.load_config()

    def compute_boxes_ci_binary(self):
        """For each simulation snapshot in the box, computes the binary contact density state of atoms."""
        # Load config
        self.load_config()

    def sample_boxes_ci_binary(self):
        """Draws samples from boxes based on the biased probability in the phi* ensemble,
        and generates a PDB with atoms colored by their contact density state."""
        # Load config
        self.load_config()

    ############################################################################
    # both calc
    ############################################################################

    def compute_boxes_ni_ci_binary(self):
        """For each simulation snapshot in the box, computes cluster membership = (binary solvation state)
        INTERSECTION (binary contact density state) state of atoms."""
        # Load config
        self.load_config()

    def sample_boxes_ni_ci_binary(self):
        """Draws samples from boxes based on the biased probability in the phi* ensemble,
        and generates a PDB with atoms colored by their cluster membership."""
        # Load config
        self.load_config()

    ############################################################################
    # computation call
    ############################################################################

    def __call__(self, calc_types):
        for calc_type in calc_types:
            if calc_type == "all":
                for f in self.func_registry.keys():
                    f()
            else:
                f = self.func_registry.get(calc_type, None)
                if f is None:
                    raise ValueError("Calc type {} not recognized.".format(calc_type))
                else:
                    f()


def main():
    parser = argparse.ArgumentParser(description='Critical cluster analysis and plots')
    parser.add_argument('config_file', help="Path to configuration file (.yaml)")
    allowed_types = list(WHAM_analysis_biasN().register().keys())
    parser.add_argument('type', nargs='+',
                        help='Types of analysis ({}) separated by space OR all'.format(",".join(allowed_types)))
    parser.add_argument('--loglevel', help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL), default=INFO', default='INFO')
    args = parser.parse_args()
    anl = CriticalClusterAnalysis(args.config_file)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    anl(args.type)


if __name__ == "__main__":
    main()
