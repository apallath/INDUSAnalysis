"""
Calculates 1D and 2D free energy profiles (in Nt and a second order parameter) for solvated polymer INDUS calculations
(biasing the solvation order parameter) using WHAM.
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


class WHAM_analysis_biasN:
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
            ("get", self.get_test_data),
            ("get2", self.get_test_data2),
            ("hist", self.plot_hist),
            ("1D", self.run_binless_log_likelihood),
            ("kappa", self.run_kappa_checks),
            ("win_KLD", self.run_reweighting_checks),
            ("phi", self.run_phi_ensemble_reweight),
            ("phi_1_star", self.run_reweight_phi_1_star),
            ("basins", self.find_basins),
            ("phi_e_star", self.run_phi_e_star_opt),
            ("phi_c_star", self.run_phi_c_star_opt),
            ("1D_boot_phi", self.run_bootstrap_ll_phi_ensemble),
            ("deltaG_diff", self.calc_deltaGu_diff_method),
            ("deltaG_int", self.calc_deltaGu_int_method_1D),
            ("2D", self.run_2D_binless_log_likelihood),
            ("sec_OP", self.run_2D_bin_sec_OP),
            ("2D_phi_stars", self.run_2D_reweight_phi_star),
            ("sec_OP_phi_stars", self.run_2D_reweight_phi_star_bin_sec_OP),
            ("2D_coex", self.run_coex_integration_2D),
            ("sec_OP_coex", self.run_coex_integration_sec_OP)
        ])
        return self.func_registry

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # set attributes
        categories = ["io_global",
                      "system",
                      "umbrellas",
                      "data_collection",
                      "1d_binning",
                      "1d_phi_ensemble",
                      "1d_phi_star",
                      "1d_plot_phi_star",
                      "find_basins",
                      "basins",
                      "coex",

                      "1d_bootstrap",

                      "2d_binning",
                      "2d_plot",
                      "2d_phi_star",
                      "2d_plot_phi_star",
                      "coex2",
                      "coexsec_OP"]
        for category in categories:
            for k, v in self.config[category].items():
                setattr(self, k, v)

    def update_config(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    def get_test_data(self):
        """
        Returns:
            tuple(n_star_win, Ntw_win, bin_points, umbrella_win, beta)

            - n_star_win: list of all simulation Nstar values for each umbrella window
            - Ntw_win: list containing an array of N~ values for each umbrella window
            - bin_points: list containing the points defining bins (centers)
                for constructing final free energy profiles
            - umbrella_win: list containing bias potential functions on N~ for each umbrella window.
            - beta: 1/kbT for the simulation, in units of mol/kJ.
        """
        # Load config
        self.load_config()

        # N* associated with each window
        # First window is unbiased
        n_star_win = ["unbiased", ]

        biased_windows = list(self.config["windows"].keys())
        biased_windows.remove("unbiased")

        n_star_win.extend(sorted(biased_windows))

        # kappa associated with each window
        kappa_win = [float(self.KAPPA)] * len(n_star_win)
        # Umbrella potentials (applied on CV) from each window
        # In this case, the potential is a harmonic potential, kappa/2 (N - N*)^2
        # For the unbiased window, the umbrella potential is 0.
        umbrella_win = [lambda x: 0]
        for i in range(1, len(n_star_win)):
            kappa = kappa_win[i]
            n_star = n_star_win[i]
            umbrella_win.append(potentials.harmonic(kappa, n_star))

        # List of bins to perform binning into
        bin_points = np.linspace(self.NMIN, self.NMAX, self.NBINS)

        # Raw, correlated timeseries CV data from each window
        Ntw_win = []

        # Read waters
        for n_star in n_star_win:
            ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(self.config["windows"][n_star]["Nt_file"])
            NTSCALE = int(self.config["windows"][n_star]["XTCDT"] / self.config["windows"][n_star]["UMBDT"])
            Ntw_win.append(ts_Ntw[self.TSTART:self.TEND:NTSCALE * self.BASE_SAMP_FREQ].data_array)
            logger.debug("(N~) N*={}: {} to end, skipping {}. {} entries.".format(n_star, self.TSTART, self.BASE_SAMP_FREQ,
                         len(ts_N[self.TSTART:self.TEND:NTSCALE * self.BASE_SAMP_FREQ].data_array)))

        beta = 1000 / (8.314 * int(self.TEMP))  # at T, in kJ/mol units

        # Show min and max Ntw across dataset
        min_Ntws = []
        for Ntwwin in Ntw_win:
            min_Ntws.append(Ntwwin.min())
        max_Ntws = []
        for Ntwwin in Ntw_win:
            max_Ntws.append(Ntwwin.max())
        logger.info("MIN Ntw = {:.2f}, MAX Ntw = {:.2f}".format(np.min(np.array(min_Ntws)), np.max(np.array(max_Ntws))))

        return n_star_win, Ntw_win, bin_points, umbrella_win, beta

    def get_test_data2(self):
        """
        Returns:
            tuple(n_star_win, Ntw_win, bin_points, umbrella_win, beta)

            - n_star_win: list of all simulation Nstar values for each umbrella window
            - Ntw_win: list containing an array of N~ values for each umbrella window
            - sec_OP_win: list containing an array of sec_OP values for each umbrella window
            - x_bin_points: list containing the points defining N~ bins (centers)
                for constructing final free energy profiles
            - y_bin_points: list containing the points defining sec_OP bins (centers)
                for constructing final free energy profiles
            - umbrella_win: list containing bias potential functions on N~ for each umbrella window.
            - beta: 1/kbT for the simulation, in units of mol/kJ.
        """
        # Load config
        self.load_config()

        # N* associated with each window
        # First window is unbiased
        n_star_win = ["unbiased", ]

        biased_windows = list(self.config["windows"].keys())
        biased_windows.remove("unbiased")

        n_star_win.extend(sorted(biased_windows))

        # kappa associated with each window
        kappa_win = [float(self.KAPPA)] * len(n_star_win)
        # Umbrella potentials (applied on CV) from each window
        # In this case, the potential is a harmonic potential, kappa/2 (N - N*)^2
        # For the unbiased window, the umbrella potential is 0.
        umbrella_win = [lambda x: 0]
        for i in range(1, len(n_star_win)):
            kappa = kappa_win[i]
            n_star = n_star_win[i]
            umbrella_win.append(potentials.harmonic(kappa, n_star))

        # List of bins to perform binning into
        x_bin_points = np.linspace(self.NMIN2, self.NMAX2, self.NBINS2)
        y_bin_points = np.linspace(self.SECOPMIN2, self.SECOPMAX2, self.SECOPBINS2)

        # Raw, correlated timeseries CV data from each window
        Ntw_win = []

        # Read waters
        for n_star in n_star_win:
            ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(self.config["windows"][n_star]["Nt_file"])
            NTSCALE = int(self.config["windows"][n_star]["XTCDT"] / self.config["windows"][n_star]["UMBDT"])
            Ntw_win.append(ts_Ntw[self.TSTART:self.TEND:NTSCALE * self.BASE_SAMP_FREQ2].data_array)
            logger.debug("(N~) N*={}: {} to end, skipping {}. {} entries.".format(n_star, self.TSTART, self.BASE_SAMP_FREQ2,
                         len(ts_N[self.TSTART:self.TEND:NTSCALE * self.BASE_SAMP_FREQ].data_array)))

        tsa = TimeSeriesAnalysis()

        sec_OP_win = []

        for n_star in n_star_win:
            ts = tsa.load_TimeSeries(self.config["windows"][n_star]["sec_OP_file"])
            sec_OP_win.append(ts[self.TSTART:self.TEND:self.BASE_SAMP_FREQ2].data_array)
            logger.debug("(sec_OP) N*={}: {} to end, skipping {}. {} entries.".format(n_star, self.TSTART, self.BASE_SAMP_FREQ2,
                         len(ts[self.TSTART:self.TEND:self.BASE_SAMP_FREQ2].data_array)))

        beta = 1000 / (8.314 * int(self.TEMP))  # at T K, in kJ/mol units

        # Show min and max Ntw across dataset
        min_Ntws = []
        for Ntwwin in Ntw_win:
            min_Ntws.append(Ntwwin.min())
        max_Ntws = []
        for Ntwwin in Ntw_win:
            max_Ntws.append(Ntwwin.max())
        logger.info("MIN Ntw = {:.2f}, MAX Ntw = {:.2f}".format(np.min(np.array(min_Ntws)), np.max(np.array(max_Ntws))))

        # Show min and max sec_OP across dataset
        min_sec_OPs = []
        for sec_OPwin in sec_OP_win:
            min_sec_OPs.append(sec_OPwin.min())
        max_sec_OPs = []
        for sec_OPwin in sec_OP_win:
            max_sec_OPs.append(sec_OPwin.max())
        logger.info("MIN sec_OP = {:.2f}, MAX sec_OP = {:.2f}".format(np.min(np.array(min_sec_OPs)), np.max(np.array(max_sec_OPs))))

        return n_star_win, Ntw_win, sec_OP_win, x_bin_points, y_bin_points, umbrella_win, beta

    ############################################################################
    # Histogram
    ############################################################################

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

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        # Prepare plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        # Setup normalization and colormap
        normalize = mcolors.Normalize(vmin=n_star_win[1], vmax=n_star_win[-1])
        colormap = cm.rainbow

        base_samp_freq = self.BASE_SAMP_FREQ

        for i in range(len(n_star_win)):
            Ntw_i = Ntw_win[i]
            NTSCALE = int(self.config["windows"][n_star_win[i]]["XTCDT"] / self.config["windows"][n_star_win[i]]["UMBDT"])
            hist, edges = np.histogram(Ntw_i[self.TSTART:self.TEND:NTSCALE * base_samp_freq], bins=bin_points, density=True)
            x = 0.5 * (edges[1:] + edges[:-1])
            y = hist
            ax.plot(x, y, color=colormap(normalize(Ntw_i[self.TSTART:self.TEND:NTSCALE * base_samp_freq].mean())))
            ax.fill_between(x, 0, y, color=colormap(normalize(Ntw_i[self.TSTART:self.TEND:NTSCALE * base_samp_freq].mean())), alpha=0.4)

        ax.set_xlabel(r"$\tilde{N}$")

        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle=':')

        # Display colorbar
        scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappable.set_array(np.array(n_star_win[1:]))
        fig.colorbar(scalarmappable, label=r"$N*$")

        # Show plot
        plt.savefig(self.plotoutdir + "/" + params["hist_imgfile"], format="png", bbox_inches='tight')
        plt.close()

    ############################################################################
    ############################################################################
    ############################################################################
    # 1D
    ############################################################################
    ############################################################################
    ############################################################################

    ############################################################################
    # WHAM computation, 1D plot, and checks
    ############################################################################

    def run_binless_log_likelihood(self):
        """
        Runs 1D binless log likelihood calculation.

        Loads the following params from the config file:
            calcfile:
            betaF_datfile:
            betaF_imgfile:
            prob_datfile:
            prob_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_binless_log_likelihood"]

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        # Perform WHAM calculation
        calc = WHAM.binless.Calc1D()
        betaF_bin, betaF_bin_counts, status = calc.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                                         bin_style='center', solver='log-likelihood',
                                                                         logevery=1)  # solver kwargs
        g_i = calc.g_i

        # Save calc
        with open(self.calcoutdir + "/" + params["calcfile"], "wb") as calcf:
            pickle.dump(calc, calcf)

        # Optimized?
        logger.debug(status)

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        betaF_bin = betaF_bin - np.min(betaF_bin)  # reposition zero so that unbiased free energy is zero

        # Write to text file
        of = open(self.calcoutdir + "/" + params["betaF_datfile"], "w")
        of.write("# Nt    betaF\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], betaF_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, betaF_bin, label="Log-likelihood binless WHAM solution")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$\beta F$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["betaF_imgfile"], bbox_inches='tight')
        plt.close()

        """Probabilities"""
        delta_x_bin = bin_points[1] - bin_points[0]
        p_bin = delta_x_bin * np.exp(-betaF_bin)

        p_bin = p_bin / (delta_x_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + params["prob_datfile"], "w")
        of.write("# Nt    Pv(Nt)\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, p_bin, label="Log-likelihood probability density")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$P_v(\tilde{N})$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["prob_imgfile"], bbox_inches='tight')
        plt.close()

    def run_kappa_checks(self):
        """
        Compares kappa value against curvature at all plot points.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_kappa_checks"]

        if not params["saved"]:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()
        kappa = self.KAPPA

        saveloc = self.calcoutdir + "/" + params["in_calcfile"]
        calc = pickle.load(open(saveloc, "rb"))

        betaF, betaF_bin_counts = calc.bin_betaF_profile(bin_points, bin_style="center")
        betaF = betaF - np.min(betaF)

        betaF1 = np.gradient(betaF, bin_points)
        betaF2 = np.gradient(betaF1, bin_points)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, -betaF2, label="Curvature")
        ax.axhline(beta * kappa, label=r"$\beta\kappa$")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$-\beta \frac{d^2F_v}{d\tilde{N}^2}$")
        ax.margins(x=0, y=0)

        ax.legend()

        plt.savefig(self.plotoutdir + "/" + params["imgfile"], bbox_inches='tight')
        plt.close()

    def run_reweighting_checks(self):
        """
        Reweights 1D profile to different N* umbrellas, compares second derivatives
        of biased profiles to kappa, reports the the KL divergences between reweighted profiles and biased profiles,
        and checks that these are under a specific threshold.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            win_dir:
            win_format:
            KLD_thresh:
            KLD_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_reweighting_checks"]

        if not params["saved"]:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        saveloc = self.calcoutdir + "/" + params["in_calcfile"]
        calc = pickle.load(open(saveloc, "rb"))

        betaF_il, _ = WHAM.statistics.win_betaF(Ntw_win, bin_points, umbrella_win, beta,
                                                bin_style='center')
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, umbrella_win,
                                                                         beta, bin_style='center')

        # Check if windows path exists
        if not os.path.exists(self.plotoutdir + "/" + params["win_dir"]):
            os.makedirs(self.plotoutdir + "/" + params["win_dir"])

        for i in range(betaF_il.shape[0]):
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

            # Plot biased profiles
            betaF_shift = np.min(betaF_il[i])
            ax.plot(bin_points, betaF_il[i] - betaF_shift, 'x--', label="Window", color="C{}".format(i))
            ax.plot(bin_points, betaF_il_reweight[i] - betaF_shift, label="Reweighted", color="C{}".format(i))

            ax.set_xlim([0, self.NMAX])
            ax.set_ylim([0, 8])

            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$\beta F_{bias, i}$")

            ax.legend()

            ax.set_title(r"$N^*$ = {}".format(n_star_win[i]))

            plt.savefig(self.plotoutdir + "/" + params["win_dir"] + "/" + params["win_format"].format(n_star_win[i]), bbox_inches='tight')
            plt.close()

        # KL divergence check
        D_KL_i = WHAM.statistics.binless_KLD_reweighted_win_betaF(calc, Ntw_win, bin_points,
                                                                  umbrella_win, beta, bin_style='center')
        logger.info("KL divergences:", D_KL_i)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(n_star_win, D_KL_i, 's')
        ax.axhline(y=params["KLD_thresh"])
        ax.set_ylim([0, None])

        ax.set_xlabel(r"$N^*$")
        ax.set_ylabel(r"$D_{KL}$")

        plt.savefig(self.plotoutdir + "/" + params["KLD_imgfile"], bbox_inches='tight')
        plt.close()

        problem_i_vals = np.argwhere(D_KL_i > params["KLD_thresh"]).flatten()

        if len(problem_i_vals) > 0:
            logger.warning("Problem N* vals:", np.array(n_star_win)[problem_i_vals])

    ############################################################################
    # 1D phi-ensemble reweighting
    ############################################################################

    def run_phi_ensemble_reweight(self):
        """
        Reweights 1D profile and calculates average N~ and Var(N~) in the phi-ensemble.
        Uses averages to estimate phi_1_star.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            phi_ens_datfile:
            phi_ens_imgfile:
            phi_ens_peaks_datfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_phi_ensemble_reweight"]

        if not params["saved"]:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        saveloc = self.calcoutdir + "/" + params["in_calcfile"]
        calc = pickle.load(open(saveloc, "rb"))

        phi_vals = np.linspace(self.PHI_BIN_MIN, self.PHI_BIN_MAX, self.PHI_BINS)

        N_avg_vals, N_var_vals = WHAM.statistics.binless_reweight_phi_ensemble(calc, phi_vals, beta)

        peaks, _ = find_peaks(N_var_vals, height=self.PEAK_CUT)

        fig, ax = plt.subplots(2, 1, figsize=(4, 8), dpi=150)
        ax[0].plot(beta * phi_vals, N_avg_vals)
        ax[0].plot(beta * phi_vals[peaks], N_avg_vals[peaks], 'x')
        ax[0].set_xlabel(r"$\beta \phi$")
        ax[0].set_ylabel(r"$\langle \tilde{N} \rangle_\phi$")

        dx = phi_vals[1] - phi_vals[0]
        dydx = np.gradient(N_avg_vals, dx)

        ax[1].plot(beta * phi_vals, N_var_vals, label=r"$\langle \delta \tilde{N}^2 \rangle_\phi$", color="C0")
        ax[1].plot(beta * phi_vals, -1 / beta * dydx, label=r"$-\frac{1}{\beta} \frac{\partial \tilde{N}}{\partial \phi}$", color="C2")
        ax[1].set_xlabel(r"$\beta \phi$")
        ax[1].set_ylabel(r"$\langle \delta \tilde{N}^2 \rangle_\phi$")
        ax[1].legend()

        ax[0].margins(x=0)
        ax[1].margins(x=0)

        logger.debug(phi_vals[np.argmax(N_var_vals)])

        sorted_peaks = sorted(peaks)
        for i, peak in enumerate(sorted_peaks):
            logger.debug(beta * phi_vals[peak])
            ax[1].text(beta * phi_vals[peak], N_var_vals[peak], r"$\beta \phi_{}* = {:.3f}$".format(i + 1, beta * phi_vals[peak]))

        plt.savefig(self.plotoutdir + "/" + params["phi_ens_imgfile"], bbox_inches='tight')
        plt.close()

        # Write to text file
        of = open(self.calcoutdir + "/" + params["phi_ens_datfile"], "w")
        of.write("# phi    <N>    <dN^2>\n")
        for i in range(len(phi_vals)):
            of.write("{:.5f} {:.5f} {:.5f}\n".format(phi_vals[i], N_avg_vals[i], N_var_vals[i]))
        of.close()

        # Write peak information to text file
        of = open(self.calcoutdir + "/" + params["phi_ens_peaks_datfile"], "w")
        of.write("# phi    beta*phi\n")
        for i, peak in enumerate(sorted(peaks)):
            of.write("{:.5f} {:.5f}\n".format(phi_vals[peak], beta * phi_vals[peak]))
        of.close()

        self.config["1d_phi_star"]["PHI_STAR"] = float("{:.5f}".format(sorted(phi_vals[peaks])[0]))
        self.config["2d_phi_star"]["PHI_STAR2"] = float("{:.5f}".format(sorted(phi_vals[peaks])[0]))
        self.config["1d_phi_star"]["PHI_STAR_EQ"] = float("{:.5f}".format(sorted(phi_vals[peaks])[0]))
        self.config["2d_phi_star"]["PHI_STAR_EQ2"] = float("{:.5f}".format(sorted(phi_vals[peaks])[0]))
        self.config["1d_phi_star"]["PHI_STAR_COEX"] = float("{:.5f}".format(sorted(phi_vals[peaks])[0]))
        self.config["2d_phi_star"]["PHI_STAR_COEX2"] = float("{:.5f}".format(sorted(phi_vals[peaks])[0]))
        self.update_config()

    ############################################################################
    # phi_1* reweighting
    ############################################################################

    def run_reweight_phi_1_star(self):
        """
        Reweights 1D profile to phi_1* ensemble.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            betaFrew_datfile:
            betaFrew_imgfile:
            probrew_datfile:
            probrew_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_reweight_phi_1_star"]

        if not params["saved"]:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        saveloc = self.calcoutdir + "/" + params["in_calcfile"]
        calc = pickle.load(open(saveloc, "rb"))

        phi_1_star = self.PHI_STAR
        logger.debug(phi_1_star)

        umb_phi_1_star = potentials.linear(phi_1_star)
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, [umb_phi_1_star],
                                                                         beta, bin_style='center')
        betaF_rew = betaF_il_reweight[0]
        betaF_rew = betaF_rew - np.min(betaF_rew)

        # Write to text file
        of = open(self.calcoutdir + "/" + params["betaFrew_datfile"], "w")
        of.write("# N    betaF\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], betaF_rew[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        indices = np.where(np.logical_and(bin_points >= self.PLOT_PHI_STAR_N_MIN,
                                          bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
        ax.plot(bin_points[indices], betaF_rew[indices], label=r"Biased free energy profile in $\phi_1^*$ ensemble.")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$\beta F$")
        ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX])

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_1_star))

        plt.savefig(self.plotoutdir + "/" + params["betaFrew_imgfile"], bbox_inches='tight')
        plt.close()

        """Probabilities"""
        delta_x_bin = bin_points[1] - bin_points[0]
        p_bin = delta_x_bin * np.exp(-betaF_rew)

        p_bin = p_bin / (delta_x_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + params["probrew_datfile"], "w")
        of.write("# Nt    Pv(Nt)\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, p_bin, label=r"Log-likelihood probability density in $\phi_1^*$ ensemble.")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$P_v(\tilde{N})$")

        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])

        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["probrew_imgfile"], bbox_inches='tight')
        plt.close()

    ############################################################################
    # phi_e* optimization
    ############################################################################

    def find_basins(self):
        """
        Finds collapsed basin (NC) and extended basin (NE) in N~ profile.
        """
        # Load config
        self.load_config()

        # load phi_1* ensemble data
        nt = []
        fnt = []
        with open(self.calcoutdir + "/" + self.config["func_params"]["run_reweight_phi_1_star"]["betaFrew_datfile"]) as f:
            for line in f:
                if line.strip().split()[0] != '#':
                    nt.append(float(line.strip().split()[0]))
                    fnt.append(float(line.strip().split()[1]))
            f.close()

        nt = np.array(nt)
        fnt = np.array(fnt)

        # indices to search over
        nc_indices = np.where(np.logical_and(nt >= self.NC_MIN,
                                             nt <= self.NC_MAX))[0]
        ne_indices = np.where(np.logical_and(nt >= self.NE_MIN,
                                             nt <= self.NE_MAX))[0]

        nc_idx = np.argmin(fnt[nc_indices])
        ne_idx = np.argmin(fnt[ne_indices])

        # Write to config file
        self.config["basins"]["NC"] = int(nt[nc_indices][nc_idx])
        self.config["basins"]["NE"] = int(nt[ne_indices][ne_idx])

        self.update_config()

    def reweight_get_basin_diff2(self, phi, calc, bin_points, beta):
        """
        Objective function to minimize to calculate phi_e*
        """
        umb_phi = potentials.linear(phi)
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, [umb_phi],
                                                                         beta, bin_style='center')
        betaF_rew = betaF_il_reweight[0]
        betaF_rew = betaF_rew - np.min(betaF_rew)
        ne_idx = np.argmin(np.abs(bin_points - self.NE))
        nc_idx = np.argmin(np.abs(bin_points - self.NC))
        objective = (betaF_rew[ne_idx] - betaF_rew[nc_idx]) ** 2
        logger.debug("phi = {:.5f}, betaphi = {:.5f}, objective = {:.5f}".format(float(phi),
                                                                                 beta * float(phi),
                                                                                 float(objective)))
        return objective

    def run_phi_e_star_opt(self):
        """
        Finds optimal phi_e* and reweights 1D profile to phi_e* ensemble.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            opt_thresh:
            betaFrew_datfile:
            betaFrew_imgfile:
            probrew_datfile:
            probrew_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_phi_e_star_opt"]

        if not params["saved"]:
            raise RuntimeError("Saved data required.")

        # Load params
        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()
        saveloc = self.calcoutdir + "/" + params["in_calcfile"]
        calc = pickle.load(open(saveloc, "rb"))

        # Optimize
        res = scipy.optimize.minimize(self.reweight_get_basin_diff2,
                                      self.PHI_STAR_EQ,
                                      args=(calc, bin_points, beta),
                                      method='Nelder-Mead',
                                      tol=params["opt_thresh"])

        phi_e_star = float(res.x)

        # Save results
        self.config["1d_phi_star"]["PHI_STAR_EQ"] = float("{:.5f}".format(phi_e_star))
        self.config["2d_phi_star"]["PHI_STAR_EQ2"] = float("{:.5f}".format(phi_e_star))
        self.update_config()

        # Plot
        umb_phi_e_star = potentials.linear(phi_e_star)
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, [umb_phi_e_star],
                                                                         beta, bin_style='center')
        betaF_rew = betaF_il_reweight[0]
        betaF_rew = betaF_rew - np.min(betaF_rew)

        # Write to text file
        of = open(self.calcoutdir + "/" + params["betaFrew_datfile"], "w")
        of.write("# N    betaF\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], betaF_rew[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        indices = np.where(np.logical_and(bin_points >= self.PLOT_PHI_STAR_N_MIN,
                                          bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
        ax.plot(bin_points[indices], betaF_rew[indices], label=r"Biased free energy profile in $\phi_e^*$ ensemble.")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$\beta F$")
        ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX])

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_e_star))

        plt.savefig(self.plotoutdir + "/" + params["betaFrew_imgfile"], bbox_inches='tight')
        plt.close()

        """Probabilities"""
        delta_x_bin = bin_points[1] - bin_points[0]
        p_bin = delta_x_bin * np.exp(-betaF_rew)

        p_bin = p_bin / (delta_x_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + params["probrew_datfile"], "w")
        of.write("# Nt    Pv(Nt)\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, p_bin, label=r"Log-likelihood probability density in $\phi_e^*$ ensemble.")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$P_v(\tilde{N})$")

        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])

        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["probrew_imgfile"], bbox_inches='tight')
        plt.close()

    ############################################################################
    # deltaGu calculation by basin differences
    ############################################################################

    def calc_deltaGu_diff_method(self):
        """
        Calculates difference between N_C and N_E points to get deltaGu.
        If saved bootstrap errors are available, uses these to get error bars on deltaGu.

        Loads the following params from the config file:
            boot_errors:
            deltaGu_datfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["calc_deltaGu_diff_method"]

        f = open(self.calcoutdir + "/" + self.config["func_params"]["run_binless_log_likelihood"]["betaF_datfile"])

        nt = []
        fnt = []

        for line in f:
            if line.strip().split()[0] != '#':
                nt.append(float(line.strip().split()[0]))
                fnt.append(float(line.strip().split()[1]))

        f.close()

        nt = np.array(nt)
        fnt = np.array(fnt)

        # Delta G
        ne_idx = np.argmin(np.abs(nt - self.NE))
        nc_idx = np.argmin(np.abs(nt - self.NC))

        deltaGu = fnt[ne_idx] - fnt[nc_idx]

        deltaGuerr = 0

        if params["boot_errors"]:
            f = open(self.calcoutdir + "/" + self.config["func_params"]["run_bootstrap_ll_phi_ensemble"]["betaFboot_datfile"])

            nt = []
            fnt = []
            fnterr = []

            for line in f:
                if line.strip().split()[0] != '#':
                    nt.append(float(line.strip().split()[0]))
                    fnt.append(float(line.strip().split()[1]))
                    fnterr.append(float(line.strip().split()[2]))
            f.close()

            nt = np.array(nt)
            fnt = np.array(fnt)

            # Delta G
            ne_idx = np.argmin(np.abs(nt - self.NE))
            nc_idx = np.argmin(np.abs(nt - self.NC))

            deltaGuerr = fnterr[ne_idx] + fnterr[nc_idx]

        # Write to text file
        of = open(self.calcoutdir + "/" + params["deltaGu_datfile"], "w")
        of.write("{:.5f} {:.5f}\n".format(deltaGu, deltaGuerr))
        of.close()

        return deltaGu, deltaGuerr

    ############################################################################
    # phi_c* optimization
    ############################################################################

    def reweight_get_basin_int2(self, phi, calc, bin_points, beta):
        """
        Objective function to minimize to calculate phi_c*
        """
        umb_phi = potentials.linear(phi)
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, [umb_phi],
                                                                         beta, bin_style='center')
        betaF_rew = betaF_il_reweight[0]
        betaF_rew = betaF_rew - np.min(betaF_rew)

        idxC = np.argwhere(bin_points < self.NT_SPLIT)
        idxE = np.argwhere(bin_points >= self.NT_SPLIT)

        gC = -logsumexp(-betaF_rew[idxC].flatten())
        gE = -logsumexp(-betaF_rew[idxE].flatten())

        objective = (gC - gE) ** 2
        logger.debug("phi = {:.5f}, betaphi = {:.5f}, objective = {:.5f}".format(float(phi),
                                                                                 beta * float(phi),
                                                                                 float(objective)))
        return objective

    def run_phi_c_star_opt(self, plot=True):
        """
        Finds optimal phi_e* and reweights 1D profile to phi_e* ensemble.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            opt_thresh:
            betaFrew_datfile:
            betaFrew_imgfile:
            probrew_datfile:
            probrew_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_phi_c_star_opt"]

        if not params["saved"]:
            raise RuntimeError("Saved data required.")

        # Load params
        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()
        saveloc = self.calcoutdir + "/" + params["in_calcfile"]
        calc = pickle.load(open(saveloc, "rb"))

        # Optimize
        res = scipy.optimize.minimize(self.reweight_get_basin_int2,
                                      self.PHI_STAR_COEX,
                                      args=(calc, bin_points, beta),
                                      method='Nelder-Mead',
                                      tol=params["opt_thresh"])

        phi_c_star = float(res.x)

        # Save results
        self.config["1d_phi_star"]["PHI_STAR_COEX"] = float("{:.5f}".format(phi_c_star))
        self.config["2d_phi_star"]["PHI_STAR_COEX2"] = float("{:.5f}".format(phi_c_star))
        self.update_config()

        # Compute profile at phi_c*
        umb_phi = potentials.linear(phi_c_star)
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, [umb_phi],
                                                                         beta, bin_style='center')
        betaF_rew = betaF_il_reweight[0]
        betaF_rew = betaF_rew - np.min(betaF_rew)

        idxC = np.argwhere(bin_points < self.NT_SPLIT)
        idxE = np.argwhere(bin_points >= self.NT_SPLIT)

        delta_x_bin = bin_points[1] - bin_points[0]
        p_bin = delta_x_bin * np.exp(-betaF_rew)
        p_bin = p_bin / (delta_x_bin * np.sum(p_bin))  # normalize

        pC = np.sum(p_bin[idxC].flatten())
        pE = np.sum(p_bin[idxE].flatten())

        # Write to text file
        of = open(self.calcoutdir + "/" + params["betaFrew_datfile"], "w")
        of.write("# N    betaF\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], betaF_rew[i]))
        of.close()

        # Plot results
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        indices = np.where(np.logical_and(bin_points >= self.PLOT_PHI_STAR_N_MIN,
                                          bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
        ax.plot(bin_points[indices], betaF_rew[indices], label=r"Biased free energy profile in $\phi_c^*$ ensemble.")
        ax.axvline(x=self.NT_SPLIT, label=r"$\tilde{{N}}={:.2f}$".format(self.NT_SPLIT))
        ax.text(0.2, 0.5, "P = {:.2f}".format(pC), transform=ax.transAxes)
        ax.text(0.8, 0.5, "P = {:.2f}".format(pE), transform=ax.transAxes)
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$P_v(\tilde{N})$")
        ax.legend()
        ax.margins(x=0, y=0)
        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])

        plt.savefig(self.plotoutdir + "/" + "coex_Nt.png", bbox_inches='tight')
        plt.close()

        """Probabilities"""
        # Write to text file
        of = open(self.calcoutdir + "/" + params["probrew_datfile"], "w")
        of.write("# Nt    Pv(Nt)\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, p_bin, label=r"Log-likelihood probability density in $\phi_c^*$ ensemble.")
        ax.axvline(x=self.NT_SPLIT, label=r"$\tilde{{N}}={:.2f}$".format(self.NT_SPLIT))
        ax.text(0.2, 0.5, "P = {:.2f}".format(pC), transform=ax.transAxes)
        ax.text(0.8, 0.5, "P = {:.2f}".format(pE), transform=ax.transAxes)
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$P_v(\tilde{N})$")
        ax.legend()
        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["probrew_imgfile"], bbox_inches='tight')
        plt.close()

    ############################################################################
    # deltaGu calculation by basin integrations
    ############################################################################

    def calc_deltaGu_int_method_1D(self):
        """Calculate ratio of C basin to E basin probabilities, in N~, to get deltaGu."""
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["calc_deltaGu_int_method_1D"]

        f = open(self.calcoutdir + "/" + self.config["func_params"]["run_binless_log_likelihood"]["betaF_datfile"])

        nt = []
        fnt = []

        for line in f:
            if line.strip().split()[0] != '#':
                nt.append(float(line.strip().split()[0]))
                fnt.append(float(line.strip().split()[1]))

        f.close()

        nt = np.array(nt)
        fnt = np.array(fnt)

        idx1 = np.argwhere(nt < self.NT_SPLIT)
        idx2 = np.argwhere(nt >= self.NT_SPLIT)

        g1 = -logsumexp(-fnt[idx1].flatten())
        g2 = -logsumexp(-fnt[idx2].flatten())

        deltaGu = g2 - g1

        # Write to text file
        of = open(self.calcoutdir + "/" + params["deltaGu_datfile"], "w")
        of.write("{:.5f}\n".format(deltaGu))
        of.close()

        return deltaGu

    ############################################################################
    # 1D phi-ensemble bootstrapping and phi_1_star + error bar calculation
    ############################################################################

    def boot_worker(self, boot_worker_idx):
        """Picklable bootstrap worker"""
        # Get data
        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()
        np.random.seed((16091 + 17 * boot_worker_idx) % 824633720831)
        Ntw_win_boot = timeseries.bootstrap_window_samples(Ntw_win)
        logger.info("Worker {}: N* = {}".format(boot_worker_idx, [len(win) for win in Ntw_win_boot]))

        phi_vals = np.linspace(self.PHI_BIN_MIN, self.PHI_BIN_MAX, self.PHI_BINS)

        # Perform WHAM calculation
        calc = WHAM.binless.Calc1D()
        betaF_bin, betaF_bin_counts, status = calc.compute_betaF_profile(Ntw_win_boot, bin_points, umbrella_win, beta,
                                                                         bin_style='center', solver='log-likelihood')  # solver kwargs

        betaF_bin = betaF_bin - np.min(betaF_bin)  # reposition zero so that unbiased free energy is zero

        # Perform phi-ensemble reweighting
        N_avg_vals, N_var_vals = WHAM.statistics.binless_reweight_phi_ensemble(calc, phi_vals, beta)

        dx = phi_vals[1] - phi_vals[0]
        dydx = np.gradient(N_avg_vals, dx)
        susc = -1 / beta * dydx

        # Calculate delta_G_fold (TODO)

        # Return
        return {"betaF": betaF_bin,
                "Navg": N_avg_vals,
                "Nvar": N_var_vals,
                "Nsusc": susc}

    def run_bootstrap_ll_phi_ensemble(self):
        """
        Runs 1D binless log likelihood calculation and phi-ensemble reweighting.

        Loads the following params from the config file:
            betaFboot_datfile:
            betaFboot_imgfile:
            phi_ens_boot_datfile:
            phi_ens_boot_imgfile:
            phi_ens_peaks_boot_datfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_bootstrap_ll_phi_ensemble"]

        with Pool(processes=self.NWORKERS) as pool:
            ret_dicts = pool.map(self.boot_worker, range(self.NBOOT))

        # Unpack returned values, calculate error bars, etc
        betaF_all = []
        N_avg_all = []
        N_var_all = []
        for boot_idx in range(self.NBOOT):
            betaF_all.append(ret_dicts[boot_idx]["betaF"])
            N_avg_all.append(ret_dicts[boot_idx]["Navg"])
            N_var_all.append(ret_dicts[boot_idx]["Nvar"])

        betaF_all = np.array(betaF_all)
        logger.debug(betaF_all.shape)
        N_avg_all = np.array(N_avg_all)
        logger.debug(N_avg_all.shape)
        N_var_all = np.array(N_var_all)
        logger.debug(N_var_all.shape)

        betaF = betaF_all.mean(axis=0)
        betaF_err = betaF_all.std(axis=0)

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        # Write to text file
        of = open(self.calcoutdir + "/" + params["betaFboot_datfile"], "w")
        of.write("# Nt    betaF    sem(betaF)\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f} {:.5f}\n".format(bin_points[i], betaF[i], betaF_err[i]))
        of.close()

        # Plot free energy profiles
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, betaF, label="Average")
        ax.fill_between(bin_points, betaF - betaF_err, betaF + betaF_err, alpha=0.5, label="Error")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$\beta F$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["betaFboot_imgfile"], bbox_inches='tight')
        plt.close()

        N_avg = N_avg_all.mean(axis=0)
        N_avg_err = N_avg_all.std(axis=0)

        N_var = N_var_all.mean(axis=0)
        N_var_err = N_var_all.std(axis=0)

        phi_vals = np.linspace(self.PHI_BIN_MIN, self.PHI_BIN_MAX, self.PHI_BINS)

        # Write to text file
        of = open(self.calcoutdir + "/" + params["phi_ens_boot_datfile"], "w")
        of.write("# phi    <N>    sem(N)   <dN^2>    sem(dN^2)\n")
        for i in range(len(phi_vals)):
            of.write("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(phi_vals[i], N_avg[i], N_avg_err[i],
                                                                   N_var[i], N_var_err[i]))
        of.close()

        phi_1_stars = np.zeros(self.NBOOT)
        phi_2_stars = np.zeros(self.NBOOT)

        # START
        all_peaks = []

        # Calculate phi_1_star and error bars on phi_1_star from bootstrapping
        for nb in range(self.NBOOT):
            # Find peak indices
            peaks, _ = find_peaks(N_var_all[nb, :], height=self.PEAK_CUT)

            # Sort peak heights
            peak_heights = N_var_all[nb, peaks]

            # Sort peaks by peak heights: largest two peaks are phi_1* and phi_2*
            sort_order = np.argsort(peak_heights)[::-1]
            peaks = peaks[sort_order]

            all_peaks.append(phi_vals[peaks])

            # Smaller index is phi_1*, larger index is phi_2*
            phi_1_stars[nb] = phi_vals[min(peaks[0], peaks[1])]
            phi_2_stars[nb] = phi_vals[max(peaks[0], peaks[1])]

        of = open(self.calcoutdir + "/" + params["phi_ens_peaks_boot_datfile"], "w")

        of.write("Peak phi values:\n")
        for phi_peaks in all_peaks:
            of.write(" ".join(["{:.5f}".format(peak) for peak in beta * phi_peaks]) + "\n")
        of.write("\n")

        of.write("beta phi_1* and beta phi_2* values:\n")
        for nb in range(self.NBOOT):
            of.write("{:.5f} {:.5f}\n".format(beta * phi_1_stars[nb], beta * phi_2_stars[nb]))
        of.write("\n")

        # Plot phi-ensemble
        phi_1_star_avg = phi_1_stars.mean()
        phi_1_star_std = phi_1_stars.std()

        phi_2_star_avg = phi_2_stars.mean()
        phi_2_star_std = phi_2_stars.std()

        peak_1_idx = (np.abs(phi_vals - phi_1_star_avg)).argmin()
        peak_2_idx = (np.abs(phi_vals - phi_2_star_avg)).argmin()

        # average peaks
        peaks = np.array([peak_1_idx, peak_2_idx])
        #END

        fig, ax = plt.subplots(2, 1, figsize=(4, 8), dpi=300)
        ax[0].plot(beta * phi_vals, N_avg)
        ax[0].fill_between(beta * phi_vals, N_avg - N_avg_err, N_avg + N_avg_err, alpha=0.5)
        ax[0].plot(beta * phi_vals[peaks], N_avg[peaks], 'x')
        ax[0].set_xlabel(r"$\beta \phi$")
        ax[0].set_ylabel(r"$\langle \tilde{N} \rangle_\phi$")

        ax[1].plot(beta * phi_vals, N_var)
        ax[1].plot(beta * phi_vals[peaks], N_var[peaks], 'x')
        ax[1].fill_between(beta * phi_vals, N_var - N_var_err, N_var + N_var_err, alpha=0.5)
        ax[1].set_xlabel(r"$\beta \phi$")
        ax[1].set_ylabel(r"$\langle \delta \tilde{N}^2 \rangle_\phi$")

        of.write("phi_1* = {:.5f} +- {:.5f}\n".format(beta * phi_1_star_avg, beta * phi_1_star_std))
        of.write("phi_2* = {:.5f} +- {:.5f}\n".format(beta * phi_2_star_avg, beta * phi_2_star_std))

        peak_1_idx = (np.abs(phi_vals - phi_1_star_avg)).argmin()
        peak_2_idx = (np.abs(phi_vals - phi_2_star_avg)).argmin()

        ax[1].text(beta * phi_1_star_avg, N_var[peak_1_idx], r"$\beta \phi_1^* = {:.3f} \pm {:.3f}$".format(beta * phi_1_star_avg, beta * phi_1_star_std))
        ax[1].text(beta * phi_2_star_avg, N_var[peak_2_idx], r"$\beta \phi_2^* = {:.3f} \pm {:.3f}$".format(beta * phi_2_star_avg, beta * phi_2_star_std))

        ax[0].margins(x=0)
        ax[1].margins(x=0)

        plt.savefig(self.plotoutdir + "/" + params["phi_ens_boot_imgfile"], bbox_inches='tight')
        plt.close()

        of.close()

    ############################################################################
    ############################################################################
    ############################################################################
    # 2D
    ############################################################################
    ############################################################################
    ############################################################################

    ############################################################################
    # 2D WHAM plot and sec_OP plot
    ############################################################################

    def run_2D_binless_log_likelihood(self):
        """
        Runs 2D binless log likelihood calculation.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            calcfile:
            betaF_imgfile:
            x_bins_npyfile:
            y_bins_npyfile:
            betaF_npyfile:
            prob_imgfile:
            prob_npyfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_2D_binless_log_likelihood"]

        savedloc = self.calcoutdir + "/" + params["in_calcfile"]

        n_star_win, Ntw_win, sec_OP_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

        assert(len(Ntw_win[0]) == len(sec_OP_win[0]))
        assert(len(Ntw_win[1]) == len(sec_OP_win[1]))

        # Unroll Ntw_win into a single array
        x_l = Ntw_win[0]
        for i in range(1, len(Ntw_win)):
            x_l = np.hstack((x_l, Ntw_win[i]))

        # Unroll sec_OP_win into a single array
        y_l = sec_OP_win[0]
        for i in range(1, len(sec_OP_win)):
            y_l = np.hstack((y_l, sec_OP_win[i]))

        N_i = np.array([len(arr) for arr in Ntw_win])

        if params["saved"]:
            calc = pickle.load(open(savedloc, "rb"))
        else:
            # Perform WHAM calculations
            calc = WHAM.binless.Calc1D()
            status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                                solver='log-likelihood',
                                                logevery=1)
            pickle.dump(calc, open(self.calcoutdir + "/" + params["calcfile"], "wb"))

            # Optimized?
            logger.debug(status)

        g_i = calc.g_i

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        betaF_2D_bin, (betaF_bin_counts, _1, _2) = calc.bin_2D_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                                             x_bin_style='center', y_bin_style='center')
        betaF_2D_bin = betaF_2D_bin - np.min(betaF_2D_bin)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, self.PLOT_BETAF_MAX2, self.PLOT_BETAF_LEVELS2)
        cmap = cm.RdYlBu
        contour_filled = ax.contourf(x_bin_points, y_bin_points, betaF_2D_bin.T, levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
        ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(contour_filled, cax=cax, orientation='vertical')

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$OP_2$")

        cax.set_title(r"$\beta G_v(\tilde{N}, OP_2)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.margins(x=0, y=0)

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        plt.savefig(self.plotoutdir + "/" + params["betaF_imgfile"], bbox_inches='tight')
        plt.close()

        # Write bin points to npy files
        np.save(self.calcoutdir + "/" + params["x_bins_npyfile"], x_bin_points)
        np.save(self.calcoutdir + "/" + params["y_bins_npyfile"], y_bin_points)

        # Write to npy file
        np.save(self.calcoutdir + "/" + params["betaF_npyfile"], betaF_2D_bin)

        """Probabilities"""
        delta_x_bin = x_bin_points[1] - x_bin_points[0]
        delta_y_bin = y_bin_points[1] - y_bin_points[0]
        p_bin = delta_x_bin * delta_y_bin * np.exp(-betaF_2D_bin)

        p_bin = p_bin / (delta_x_bin * delta_y_bin * np.sum(p_bin))  # normalize

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, np.max(p_bin), self.PLOT_PV_LEVELS2)
        cmap = cm.YlGnBu
        contour_filled = ax.contourf(x_bin_points, y_bin_points, p_bin.T, levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
        ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(contour_filled, cax=cax, orientation='vertical', format='%.1e')

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$OP_2$")

        cax.set_title(r"$P_v(\tilde{N}, OP_2)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.margins(x=0, y=0)

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        plt.savefig(self.plotoutdir + "/" + params["prob_imgfile"], bbox_inches='tight')
        plt.close()

        # Write to npy file
        np.save(self.calcoutdir + "/" + params["prob_npyfile"], p_bin)

    def run_2D_bin_sec_OP(self):
        """
        Loads the following params from the config file:
            saved:
            in_calcfile:
            betaF_datfile:
            betaF_imgfile:
            prob_datfile:
            prob_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_2D_bin_sec_OP"]

        savedloc = self.calcoutdir + "/" + params["in_calcfile"]

        n_star_win, Ntw_win, sec_OP_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

        assert(len(Ntw_win[0]) == len(sec_OP_win[0]))
        assert(len(Ntw_win[1]) == len(sec_OP_win[1]))

        # Unroll Ntw_win into a single array
        x_l = Ntw_win[0]
        for i in range(1, len(Ntw_win)):
            x_l = np.hstack((x_l, Ntw_win[i]))

        # Unroll sec_OP_win into a single array
        y_l = sec_OP_win[0]
        for i in range(1, len(sec_OP_win)):
            y_l = np.hstack((y_l, sec_OP_win[i]))

        if params["saved"]:
            calc = pickle.load(open(savedloc, "rb"))
        else:
            raise RuntimeError("Run WHAM calc first.")

        g_i = calc.g_i

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        betaF_sec_OP = calc.bin_second_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                 x_bin_style='center', y_bin_style='center')
        betaF_sec_OP = betaF_sec_OP - np.min(betaF_sec_OP)

        # Write to text file
        of = open(self.calcoutdir + "/" + params["betaF_datfile"], "w")
        of.write("# Nt    betaF\n")
        for i in range(len(y_bin_points)):
            of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], betaF_sec_OP[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(y_bin_points, betaF_sec_OP, label="Log-likelihood binless sec_OP profile")
        ax.set_xlabel(r"$OP_2$")
        ax.set_ylabel(r"$\beta F$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["betaF_imgfile"], bbox_inches='tight')
        plt.close()

        """Probabilities"""
        delta_y_bin = y_bin_points[1] - y_bin_points[0]
        p_bin = delta_y_bin * np.exp(-betaF_sec_OP)

        p_bin = p_bin / (delta_y_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + params["prob_datfile"], "w")
        of.write("# Nt    Pv(sec_OP)\n")
        for i in range(len(y_bin_points)):
            of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(y_bin_points, p_bin, label="Log-likelihood probability density")
        ax.set_xlabel(r"$OP_2$")
        ax.set_ylabel(r"$P_v(OP_2)$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + params["prob_imgfile"], bbox_inches='tight')
        plt.close()

    ############################################################################
    # 2D phi_1* reweighting
    ############################################################################

    def run_2D_reweight_phi_star(self):
        """
        Reweights 2D profile to different phi_star ensembles.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            betaF_imgformat:
            x_bins_npyformat:
            y_bins_npyformat:
            betaF_npyformat:
            prob_imgformat:
            prob_npyformat:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_2D_reweight_phi_star"]

        # Loop over params
        for phi_star_key in ["PHI_STAR2", "PHI_STAR_EQ2", "PHI_STAR_COEX2"]:

            savedloc = self.calcoutdir + "/" + params["in_calcfile"]

            n_star_win, Ntw_win, sec_OP_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

            assert(len(Ntw_win[0]) == len(sec_OP_win[0]))
            assert(len(Ntw_win[1]) == len(sec_OP_win[1]))

            # Unroll Ntw_win into a single array
            x_l = Ntw_win[0]
            for i in range(1, len(Ntw_win)):
                x_l = np.hstack((x_l, Ntw_win[i]))

            # Unroll sec_OP_win into a single array
            y_l = sec_OP_win[0]
            for i in range(1, len(sec_OP_win)):
                y_l = np.hstack((y_l, sec_OP_win[i]))

            if params["saved"]:
                calc = pickle.load(open(savedloc, "rb"))
            else:
                raise RuntimeError("Run WHAM calc first.")

            g_i = calc.g_i

            # Useful for debugging:
            logger.debug("Window free energies: ", g_i)

            phi_star = getattr(self, phi_star_key)

            logger.debug("phi* = {}".format(phi_star))

            G_l_rew = calc.reweight(beta, u_bias=potentials.linear(phi_star))

            betaF_2D_rew, (betaF_2D_rew_bin_counts, _1, _2) = calc.bin_2D_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                                                        G_l=G_l_rew, x_bin_style='center', y_bin_style='center')
            betaF_2D_rew = betaF_2D_rew - np.min(betaF_2D_rew)

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

            levels = np.linspace(0, self.PLOT_PHI_STAR_BETAF_MAX2, self.PLOT_PHI_STAR_BETAF_LEVELS2)
            cmap = cm.RdYlBu

            x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN2, x_bin_points <= self.PLOT_PHI_STAR_N_MAX2))[0]
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)

            y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_SECOP_MIN2, y_bin_points <= self.PLOT_PHI_STAR_SECOP_MAX2))[0]
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            contour_filled = ax.contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                         betaF_2D_rew[x_min:x_max, y_min:y_max].T,
                                         levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
            ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(contour_filled, cax=cax, orientation='vertical')

            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$OP_2$")

            cax.set_title(r"$\beta G_v^{\phi_1^*}(\tilde{N}, OP_2)$")

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.margins(x=0, y=0)

            ax.tick_params(axis='both', which='both', direction='in', pad=10)

            ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_star))

            plt.savefig(self.plotoutdir + "/" + params["betaF_imgformat"].format(phi_star_key), bbox_inches='tight')
            plt.close()

            # Write free energy to npy file
            np.save(self.calcoutdir + "/" + params["betaF_npyformat"].format(phi_star_key), betaF_2D_rew)

            # Write bin points to npy files
            np.save(self.calcoutdir + "/" + params["x_bins_npyformat"].format(phi_star_key), x_bin_points)
            np.save(self.calcoutdir + "/" + params["y_bins_npyformat"].format(phi_star_key), y_bin_points)

            """Probabilities"""
            delta_x_bin = x_bin_points[1] - x_bin_points[0]
            delta_y_bin = y_bin_points[1] - y_bin_points[0]
            p_bin = delta_x_bin * delta_y_bin * np.exp(-betaF_2D_rew)

            p_bin = p_bin / (delta_x_bin * delta_y_bin * np.sum(p_bin))  # normalize

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

            levels = np.linspace(0, np.max(p_bin), self.PLOT_PHI_STAR_PV_LEVELS2)
            cmap = cm.YlGnBu

            x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN2, x_bin_points <= self.PLOT_PHI_STAR_N_MAX2))[0]
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)

            y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_SECOP_MIN2, y_bin_points <= self.PLOT_PHI_STAR_SECOP_MAX2))[0]
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            contour_filled = ax.contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                         p_bin[x_min:x_max, y_min:y_max].T,
                                         levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
            ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(contour_filled, cax=cax, orientation='vertical', format='%.1e')

            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$OP_2$")

            cax.set_title(r"$P_v^{\phi_1^*}(\tilde{N}, OP_2)$")

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.margins(x=0, y=0)

            ax.tick_params(axis='both', which='both', direction='in', pad=10)

            ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_star))

            plt.savefig(self.plotoutdir + "/" + params["prob_imgformat"].format(phi_star_key), bbox_inches='tight')
            plt.close()

            # Write probabilities to npy file
            np.save(self.calcoutdir + "/" + params["prob_npyformat"].format(phi_star_key), p_bin)

    def run_2D_reweight_phi_star_bin_sec_OP(self):
        """
        Calculates 1D profile in sec_OP at different phi_star values by integrating out
        N coordinate from 2D reweighted profile.

        Loads the following params from the config file:
            saved:
            in_calcfile:
            betaF_datformat:
            betaF_imgformat:
            prob_datformat:
            prob_imgformat:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_2D_reweight_phi_star_bin_sec_OP"]

        # Loop over params
        for phi_star_key in ["PHI_STAR2", "PHI_STAR_EQ2", "PHI_STAR_COEX2"]:

            savedloc = self.calcoutdir + "/" + params["in_calcfile"]

            n_star_win, Ntw_win, sec_OP_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

            assert(len(Ntw_win[0]) == len(sec_OP_win[0]))
            assert(len(Ntw_win[1]) == len(sec_OP_win[1]))

            # Unroll Ntw_win into a single array
            x_l = Ntw_win[0]
            for i in range(1, len(Ntw_win)):
                x_l = np.hstack((x_l, Ntw_win[i]))

            # Unroll sec_OP_win into a single array
            y_l = sec_OP_win[0]
            for i in range(1, len(sec_OP_win)):
                y_l = np.hstack((y_l, sec_OP_win[i]))

            if params["saved"]:
                calc = pickle.load(open(savedloc, "rb"))
            else:
                raise RuntimeError("Run WHAM calc first.")

            g_i = calc.g_i

            # Useful for debugging:
            logger.debug("Window free energies: ", g_i)

            phi_star = getattr(self, phi_star_key)

            logger.debug("phi* = {}".format(phi_star))

            G_l_rew = calc.reweight(beta, u_bias=potentials.linear(phi_star))

            betaF_sec_OP_rew = calc.bin_second_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                         G_l=G_l_rew, x_bin_style='center', y_bin_style='center')
            betaF_sec_OP_rew = betaF_sec_OP_rew - np.min(betaF_sec_OP_rew)

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_SECOP_MIN2,
                                              y_bin_points <= self.PLOT_PHI_STAR_SECOP_MAX2))[0]
            ax.plot(y_bin_points[indices], betaF_sec_OP_rew[indices], label=r"Biased free energy profile in $\phi_1^*$ ensemble.")
            ax.set_xlabel(r"$OP_2$")
            ax.set_ylabel(r"$\beta F$")
            ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX2])

            ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_star))

            plt.savefig(self.plotoutdir + "/" + params["betaF_imgformat"].format(phi_star_key), bbox_inches='tight')
            plt.close()

            # Write to text file
            of = open(self.calcoutdir + "/" + params["betaF_datformat"].format(phi_star_key), "w")
            of.write("# N    betaF\n")
            for i in range(len(y_bin_points)):
                of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], betaF_sec_OP_rew[i]))
            of.close()

            """Probabilities"""
            delta_y_bin = y_bin_points[1] - y_bin_points[0]
            p_bin = delta_y_bin * np.exp(-betaF_sec_OP_rew)

            p_bin = p_bin / (delta_y_bin * np.sum(p_bin))  # normalize

            # Write to text file
            of = open(self.calcoutdir + "/" + params["prob_datformat"].format(phi_star_key), "w")
            of.write("# Nt    Pv(sec_OP)\n")
            for i in range(len(y_bin_points)):
                of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], p_bin[i]))
            of.close()

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.plot(y_bin_points, p_bin, label="Log-likelihood probability density")
            ax.set_xlabel(r"$OP_2$")
            ax.set_ylabel(r"$P_v(OP_2)$")

            ax.margins(x=0, y=0)

            ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_star))

            plt.savefig(self.plotoutdir + "/" + params["prob_imgformat"].format(phi_star_key), bbox_inches='tight')
            plt.close()

    ############################################################################
    # phi_c* coexistence integrations
    ############################################################################

    def run_coex_integration_2D(self):
        """
        Integrates reweighted 2D profile, at phi_star_coex, to determine coexistence.

        Loads the following params from config file:
            in_x_bins_npyformat:
            in_y_bins_npyformat:
            in_betaF_npyformat:
            in_prob_npyformat:
            betaF_coex_imgfile:
            prob_coex_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_coex_integration_2D"]

        pvntsecop = np.load(self.calcoutdir + "/" + params["in_prob_npyformat"].format("PHI_STAR_COEX2"))
        x_bin_points = np.load(self.calcoutdir + "/" + params["in_x_bins_npyformat"].format("PHI_STAR_COEX2"))
        y_bin_points = np.load(self.calcoutdir + "/" + params["in_y_bins_npyformat"].format("PHI_STAR_COEX2"))

        xv, yv = np.meshgrid(x_bin_points, y_bin_points, indexing='ij')
        logger.debug(xv.shape)
        mask1 = yv < self.NTSECOP_SPLIT_m * (xv - self.NTSECOP_SPLIT_x0) + self.NTSECOP_SPLIT_c
        mask2 = yv >= self.NTSECOP_SPLIT_m * (xv - self.NTSECOP_SPLIT_x0) + self.NTSECOP_SPLIT_c

        dx = x_bin_points[1] - x_bin_points[0]
        dy = y_bin_points[1] - y_bin_points[0]

        p1 = dx * dy * np.sum(mask1 * pvntsecop)
        p2 = dx * dy * np.sum(mask2 * pvntsecop)

        logger.info("Probabilities in N, sec_OP: {:.5f} {:.5f}".format(p1, p2))

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, np.max(pvntsecop), self.PLOT_PHI_STAR_PV_LEVELS2)
        cmap = cm.YlGnBu

        x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN2, x_bin_points <= self.PLOT_PHI_STAR_N_MAX2))[0]
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)

        y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_SECOP_MIN2, y_bin_points <= self.PLOT_PHI_STAR_SECOP_MAX2))[0]
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        contour_filled = ax.contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                     pvntsecop[x_min:x_max, y_min:y_max].T,
                                     levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
        ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(contour_filled, cax=cax, orientation='vertical', format='%.1e')

        # Plot dividing line
        ax.plot(x_bin_points[x_indices], self.NTSECOP_SPLIT_m * (x_bin_points[x_indices] - self.NTSECOP_SPLIT_x0) + self.NTSECOP_SPLIT_c)

        # Plot Nt, sec_OP dividing lines
        ax.plot(self.NT_SPLIT, self.SECOP_SPLIT, 'x')
        ax.text(self.NT_SPLIT + 5, self.SECOP_SPLIT + .5, "({:.2f}, {:.2f})".format(self.NT_SPLIT, self.SECOP_SPLIT))

        ax.text(0.2, 0.2, "P = {:.2f}".format(p1), transform=ax.transAxes)
        ax.text(0.8, 0.8, "P = {:.2f}".format(p2), transform=ax.transAxes)

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$OP_2$")

        cax.set_title(r"$P_v^{\phi_1^*}(\tilde{N}, OP_2)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN2, self.PLOT_PHI_STAR_N_MAX2])
        ax.set_ylim([self.PLOT_PHI_STAR_SECOP_MIN2, self.PLOT_PHI_STAR_SECOP_MAX2])

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        plt.savefig(self.plotoutdir + "/" + params["prob_coex_imgfile"], bbox_inches='tight')
        plt.close()

        # Plot free energies
        fntsecop = np.load(self.calcoutdir + "/" + params["in_betaF_npyformat"].format("PHI_STAR_COEX2"))

        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, self.PLOT_PHI_STAR_BETAF_MAX2, self.PLOT_PHI_STAR_BETAF_LEVELS2)
        cmap = cm.RdYlBu

        x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN2, x_bin_points <= self.PLOT_PHI_STAR_N_MAX2))[0]
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)

        y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_SECOP_MIN2, y_bin_points <= self.PLOT_PHI_STAR_SECOP_MAX2))[0]
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        contour_filled = ax.contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                     fntsecop[x_min:x_max, y_min:y_max].T,
                                     levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
        ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(contour_filled, cax=cax, orientation='vertical')

        # Plot dividing line
        ax.plot(x_bin_points[x_indices], self.NTSECOP_SPLIT_m * (x_bin_points[x_indices] - self.NTSECOP_SPLIT_x0) + self.NTSECOP_SPLIT_c)

        # Plot Nt, sec_OP dividing lines
        ax.plot(self.NT_SPLIT, self.SECOP_SPLIT, 'x')
        ax.text(self.NT_SPLIT + 5, self.SECOP_SPLIT + .5, "({:.2f}, {:.2f})".format(self.NT_SPLIT, self.SECOP_SPLIT))

        ax.text(0.2, 0.2, "P = {:.2f}".format(p1), transform=ax.transAxes)
        ax.text(0.8, 0.8, "P = {:.2f}".format(p2), transform=ax.transAxes)

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$OP_2$")

        cax.set_title(r"$\beta G_v^{\phi_1^*}(\tilde{N}, OP_2)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN2, self.PLOT_PHI_STAR_N_MAX2])
        ax.set_ylim([self.PLOT_PHI_STAR_SECOP_MIN2, self.PLOT_PHI_STAR_SECOP_MAX2])

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        plt.savefig(self.plotoutdir + "/" + params["betaF_coex_imgfile"], bbox_inches='tight')
        plt.close()

    def run_coex_integration_sec_OP(self):
        """
        Integrates reweighted sec_OP profile, at phi_star_coex, to determine coexistence.

        Loads the following params from config file:
            in_betaF_datformat:
            in_prob_datformat:
            betaF_coex_imgfile:
            prob_coex_imgfile:
        """
        # Load config
        self.load_config()
        # Load params
        params = self.config["func_params"]["run_coex_integration_sec_OP"]

        f = open(self.calcoutdir + "/" + params["in_prob_datformat"].format("PHI_STAR_COEX2"))

        secop = []
        pvsecop = []

        for line in f:
            if line.strip().split()[0] != '#':
                secop.append(float(line.strip().split()[0]))
                pvsecop.append(float(line.strip().split()[1]))

        f.close()

        secop = np.array(secop)
        pvsecop = np.array(pvsecop)

        idx1 = np.argwhere(secop < self.SECOP_SPLIT)
        idx2 = np.argwhere(secop >= self.SECOP_SPLIT)

        dx = secop[1] - secop[0]

        p1 = dx * np.sum(pvsecop[idx1].flatten())
        p2 = dx * np.sum(pvsecop[idx2].flatten())

        logger.info("Probabilities in sec_OP: {:.5f} {:.5f}".format(p1, p2))

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(secop, pvsecop, label="Probability density")
        ax.axvline(x=self.SECOP_SPLIT, label=r"$OP_2 ={:.2f}$".format(self.SECOP_SPLIT))
        ax.text(0.2, 0.5, "P = {:.2f}".format(p1), transform=ax.transAxes)
        ax.text(0.8, 0.5, "P = {:.2f}".format(p2), transform=ax.transAxes)
        ax.set_xlabel(r"$OP_2$")
        ax.set_ylabel(r"$P_v(OP_2)$")
        ax.legend()
        ax.margins(x=0, y=0)

        ax.set_xlim([self.PLOT_PHI_STAR_SECOP_MIN2, self.PLOT_PHI_STAR_SECOP_MAX2])

        plt.savefig(self.plotoutdir + "/" + params["prob_coex_imgfile"], bbox_inches='tight')
        plt.close()

        # Plot free energies
        f = open(self.calcoutdir + "/" + params["in_betaF_datformat"].format("PHI_STAR_COEX2"))

        fsecop = []

        for line in f:
            if line.strip().split()[0] != '#':
                fsecop.append(float(line.strip().split()[1]))

        f.close()

        fsecop = np.array(fsecop)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(secop, fsecop, label="Free energy")
        ax.axvline(x=self.SECOP_SPLIT, label=r"$OP_2 ={:.2f}$".format(self.SECOP_SPLIT))
        ax.text(0.2, 0.5, "P = {:.2f}".format(p1), transform=ax.transAxes)
        ax.text(0.8, 0.5, "P = {:.2f}".format(p2), transform=ax.transAxes)
        ax.set_xlabel(r"$OP_2$")
        ax.set_ylabel(r"$F(OP_2)$")
        ax.legend()
        ax.margins(x=0, y=0)
        ax.set_xlim([self.PLOT_PHI_STAR_SECOP_MIN2, self.PLOT_PHI_STAR_SECOP_MAX2])
        ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX2])

        plt.savefig(self.plotoutdir + "/" + params["betaF_coex_imgfile"], bbox_inches='tight')
        plt.close()

    ############################################################################
    # computation call
    ############################################################################

    def __call__(self, calc_types, calc_args={}):
        for calc_type in calc_types:
            if calc_type == "all":
                for key in self.func_registry.keys():
                    f = self.func_registry[key]
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
    allowed_types = list(WHAM_analysis_biasN().register().keys())
    parser.add_argument('type', nargs='+',
                        help='Types of analysis ({}) separated by space OR all'.format(",".join(allowed_types)))
    parser.add_argument('--loglevel', help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL), default=INFO', default='INFO')
    args = parser.parse_args()
    anl = WHAM_analysis_biasN(args.config_file)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    anl(args.type)


if __name__ == "__main__":
    main()
