"""
Calculates 2D free energy profiles for solvated polymer INDUS calculations at different temperatures
(biasing the solvation order parameter) using TWHAM.
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


class TWHAM_analysis_biasN:
    """
    Class for performing calculations and plotting figures.

    All configuration data is read from and written to a YAML configuration file.

    Args:
        config_file: Path to config yaml file (default="config.yaml")

    Attributes:
        config: Dictionary containing all configuration parameters
    """

    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.register()

    def register(self):
        self.func_registry = OrderedDict([
            ("hist", self.plot_hist)
        ])
        return self.func_registry

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # set attributes
        categories = ["binning",
                      "data_collection",
                      "io_global",
                      "system"]
        for category in categories:
            for k, v in self.config[category].items():
                setattr(self, k, v)

    def update_config(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    ############################################################################
    # Histogram
    ############################################################################

    def read_energy_xvg(self, xvg_file):
        times = []
        pots = []
        with open(xvg_file, "r") as f:
            for line in f:
                if line.strip()[0] == '#' or line.strip()[0] == '@':
                    pass
                else:
                    time, pot = [float(n) for n in line.strip().split()]
                    times.append(time)
                    pots.append(pot)
        return TimeSeries(times, pots, ["Potential Energy"])

    def plot_hist(self):
        """
        Plots histogram of N~ data

        Loads the following params from the config file:
            hist_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["plot_hist"]

        # Prepare scatter plot
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        # Setup normalization and colormap
        #normalize = mcolors.Normalize(vmin=n_star_win[1], vmax=n_star_win[-1])
        #colormap = cm.rainbow

        x_all = np.array([])
        y_all = np.array([])

        for TEMP in self.TEMPS:

            n_star_win = list(self.config["windows"][TEMP].keys())

            for nstar in n_star_win:
                if nstar != 'unbiased':
                    win_options = self.config["windows"][TEMP][nstar]

                    ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(win_options["Nt_file"])
                    NTSCALE = int(win_options["EDRDT"] / win_options["UMBDT"])
                    x_pts = ts_Ntw[self.TSTART:self.TEND:NTSCALE * win_options["BASE_SAMP_FREQ"]].data_array
                    x_all = np.hstack((x_all, x_pts))

                    ts_pot = self.read_energy_xvg(win_options["pot_file"])
                    y_pts = ts_pot[self.TSTART:self.TEND:win_options["BASE_SAMP_FREQ"]].data_array
                    y_all = np.hstack((y_all, y_pts))

                    print((TEMP, nstar), (len(x_pts), len(y_pts), len(x_all), len(y_all)))

                    ax.scatter(x_pts, y_pts, s=1)

            # Show plot
            fig.savefig(self.plotoutdir + "/" + params["scatter_imgfile"], format="png", bbox_inches='tight')

        # Prepare scatter plot
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        x_bin_points = np.linspace(self.NMIN, self.NMAX, self.NBINS)
        y_bin_points = np.linspace(self.EMIN, self.EMAX, self.EBINS)
        h = ax.hist2d(x_all, y_all, (x_bin_points, y_bin_points))
        fig.colorbar(h[3])

        fig.savefig(self.plotoutdir + "/" + params["hist_imgfile"], format="png", bbox_inches='tight')

        plt.close('all')

    ############################################################################
    # computation call
    ############################################################################

    def __call__(self, calc_types, calc_args={}):
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
    parser = argparse.ArgumentParser(description='WHAM-based analysis and plots')
    parser.add_argument('config_file', help="Path to configuration file (.yaml)")
    allowed_types = list(TWHAM_analysis_biasN().register().keys())
    parser.add_argument('type', nargs='+',
                        help='Types of analysis ({}) separated by space OR all'.format(",".join(allowed_types)))
    parser.add_argument('--loglevel', help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL), default=INFO', default='INFO')
    args = parser.parse_args()
    anl = TWHAM_analysis_biasN(args.config_file)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    anl(args.type)


if __name__ == "__main__":
    main()
