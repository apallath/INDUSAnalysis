"""
Calculates 1D and 2D free energy profiles for solvated polymer INDUS calculations
(biasing the solvation order parameter) using WHAM.

@author Akash Pallath
"""
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


class WHAM_analysis_biasN:
    """
    Class for performing calculations and plotting figures.

    All configuration data is read from and written to a YAML configuration file.

    Args:
        config_file: Path to config yaml file (default="config.yaml")

    Attributes:
        config: Dictionary containing all configuration parameters

        (System parameters <- system)
        NBEAD (int):
        TEMP (float):

        (Umbrella parameters <- umbrellas)
        KAPPA (float):

        (Data collection parameters <- data_collection)
        TSTART (float):
        BASE_SAMP_FREQ (int):
        BASE_SAMP_FREQ_2 (int):

        (1D-WHAM binning parameters <- 1d_binning)
        NMIN (float):
        NMAX (float):
        NBINS (int):

        (1D phi-ensemble reweighting and peak detection parameters <- 1d_phi_ensemble)
        PHI_BIN_MIN (float):
        PHI_BIN_MAX (float):
        PHI_BINS (int):
        PEAK_CUT (float):

        (1D phi* reweighting parameters <- 1d_phi_star)
        PHI_STAR (float):
        PHI_STAR_EQ (float):
        PHI_STAR_COEX (float):

        (2D-WHAM binning parameters <- 2d_binning)
        NMIN2 (float):
        NMAX2 (float):
        NBINS2 (int):
        RGMIN2 (float):
        RGMAX2 (float):
        RGBINS2 (int):

        (2D plot parameters <- 2d_plot)
        PLOT_N_MIN (float):
        PLOT_N_MAX (float):
        PLOT_BETAF_MAX (float):
        PLOT_BETAF_LEVELS (int):
        PLOT_PV_LEVELS (int):

        (2D phi* reweighting parameters <- 2d_phi_star)
        PHI_STAR2 (float):
        PHI_STAR_EQ2 (float):
        PHI_STAR_COEX2 (float):

        (2D phi* plot parameters <- 2d_plot_phi_star)
        PLOT_PHI_STAR_N_MIN (float):
        PLOT_PHI_STAR_N_MAX (float):
        PLOT_PHI_STAR_RG_MIN (float):
        PLOT_PHI_STAR_RG_MAX (float):
        PLOT_PHI_STAR_BETAF_MAX (float):
        PLOT_PHI_STAR_BETAF_LEVELS (int):
        PLOT_PHI_STAR_PV_LEVELS (int):

        (Coexistence splits <- coex)
        NT_SPLIT (float):
        RG_SPLIT (float):
        NTRG_SPLIT_m (float):
        NTRG_SPLIT_x0 (float):
        NTRG_SPLIT_c (float):

        (Basin definitions <- basins)
        NC (float):
        NE (float):
    """

    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # set attributes
        categories = ["system",
                      "umbrellas",
                      "data_collection",
                      "1d_binning",
                      "1d_phi_ensemble",
                      "1d_phi_star",
                      "2d_binning",
                      "2d_plot",
                      "2d_phi_star",
                      "2d_plot_phi_star",
                      "coex",
                      "basins"]
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
            Ntw_win.append(ts_Ntw[self.TSTART::NTSCALE * self.BASE_SAMP_FREQ].data_array)

        beta = 1000 / (8.314 * int(self.TEMP))  # at T, in kJ/mol units

        # Show min and max Ntw across dataset
        min_Ntws = []
        for Ntwwin in Ntw_win:
            min_Ntws.append(Ntwwin.min())
        max_Ntws = []
        for Ntwwin in Ntw_win:
            max_Ntws.append(Ntwwin.max())
        print("MIN Ntw = {:.2f}, MAX Ntw = {:.2f}".format(np.min(np.array(min_Ntws)), np.max(np.array(max_Ntws))))

        return n_star_win, Ntw_win, bin_points, umbrella_win, beta

    def get_test_data2(self):
        """
        Returns:
            tuple(n_star_win, Ntw_win, bin_points, umbrella_win, beta)

            - n_star_win: list of all simulation Nstar values for each umbrella window
            - Ntw_win: list containing an array of N~ values for each umbrella window
            - Rg_win: list containing an array of Rg values for each umbrella window
            - x_bin_points: list containing the points defining N~ bins (centers)
                for constructing final free energy profiles
            - y_bin_points: list containing the points defining Rg bins (centers)
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
        y_bin_points = np.linspace(self.RGMIN2, self.RGMAX2, self.RGBINS2)

        # Raw, correlated timeseries CV data from each window
        Ntw_win = []

        # Read waters
        for n_star in n_star_win:
            ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(self.config["windows"][n_star]["Nt_file"])
            NTSCALE = int(self.config["windows"][n_star]["XTCDT"] / self.config["windows"][n_star]["UMBDT"])
            Ntw_win.append(ts_Ntw[self.TSTART::NTSCALE * self.BASE_SAMP_FREQ2].data_array)

        tsa = TimeSeriesAnalysis()

        Rg_win = []

        for n_star in n_star_win:
            ts = tsa.load_TimeSeries(self.config["windows"][n_star]["Rg_file"])
            Rg_win.append(ts[self.TSTART::self.BASE_SAMP_FREQ2].data_array)
            print("(Rg) N*={}: {} to end, skipping {}. {} entries.".format(n_star, self.TSTART, self.BASE_SAMP_FREQ2,
                  len(ts[self.TSTART::self.BASE_SAMP_FREQ2].data_array)))

        beta = 1000 / (8.314 * int(self.TEMP))  # at T K, in kJ/mol units

        # Show min and max Rg across dataset
        min_Rgs = []
        for Rgwin in Rg_win:
            min_Rgs.append(Rgwin.min())
        max_Rgs = []
        for Rgwin in Rg_win:
            max_Rgs.append(Rgwin.max())
        logger.debug("MIN Rg = {:.2f}, MAX Rg = {:.2f}".format(np.min(np.array(min_Rgs)), np.max(np.array(max_Rgs))))

        return n_star_win, Ntw_win, Rg_win, x_bin_points, y_bin_points, umbrella_win, beta

    ############################################################################
    # Histogram
    ############################################################################

    def plot_hist(self,
                  plotoutdir=".",
                  out_hist_imgfile="nstar_waters_hist.png"):
        """
        Plots histogram of N~ data
        """
        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        # Prepare plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        # Setup normalization and colormap
        normalize = mcolors.Normalize(vmin=n_star_win[1], vmax=n_star_win[-1])
        colormap = cm.rainbow

        base_samp_freq = self.BASE_SAMP_FREQ

        for i in range(len(Ntw_win)):
            Ntw_i = Ntw_win[i]

            hist, edges = np.histogram(Ntw_i[self.NTSTART::self.NTSCALE * base_samp_freq], bins=bin_points, density=True)
            x = 0.5 * (edges[1:] + edges[:-1])
            y = hist
            ax.plot(x, y, color=colormap(normalize(Ntw_i[self.NTSTART::self.NTSCALE * base_samp_freq].mean())))
            ax.fill_between(x, 0, y, color=colormap(normalize(Ntw_i[self.NTSTART::self.NTSCALE * base_samp_freq].mean())), alpha=0.4)

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
        plt.savefig(plotoutdir + "/" + out_hist_imgfile, format="png", bbox_inches='tight')
        plt.close()

    ############################################################################
    # WHAM computation, 1D plot, and checks
    ############################################################################

    def run_binless_log_likelihood(self,
                                   calcoutdir=".",
                                   plotoutdir=".",
                                   save_calcfile="calc_1D_saved.pkl",
                                   out_betaF_datfile="binless_ll.dat",
                                   out_betaF_imgfile="binless_log_likelihood.png",
                                   out_prob_datfile="binless_ll_prob.dat",
                                   out_prob_imgfile="binless_ll_prob.png"):
        """
        Runs 1D binless log likelihood calculation.
        """
        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        # Perform WHAM calculation
        calc = WHAM.binless.Calc1D()
        betaF_bin, betaF_bin_counts, status = calc.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                                         bin_style='center', solver='log-likelihood',
                                                                         logevery=1)  # solver kwargs
        g_i = calc.g_i

        # Save calc
        with open(calcoutdir + "/" + save_calcfile, "wb") as calcf:
            pickle.dump(calc, calcf)

        # Optimized?
        logger.debug(status)

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        betaF_bin = betaF_bin - np.min(betaF_bin)  # reposition zero so that unbiased free energy is zero

        # Write to text file
        of = open(calcoutdir + "/" + out_betaF_datfile, "w")
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

        plt.savefig(plotoutdir + "/" + out_betaF_imgfile, bbox_inches='tight')
        plt.close()

        """Probabilities"""
        delta_x_bin = bin_points[1] - bin_points[0]
        p_bin = delta_x_bin * np.exp(-betaF_bin)

        p_bin = p_bin / (delta_x_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(calcoutdir + "/" + out_prob_datfile, "w")
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

        plt.savefig(plotoutdir + "/" + out_prob_imgfile, bbox_inches='tight')
        plt.close()

    def run_kappa_checks(self,
                         saved=True,
                         calcindir=".",
                         savefile="calc_1D_saved.pkl"):
        if not saved:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()
        kappa = self.get_kappa()

        saveloc = self.calcoutdir + "/" + savefile
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

        plt.savefig(self.plotoutdir + "/" + "curvature_kappa.png", bbox_inches='tight')
        plt.close()

    def run_reweighting_checks(self, saved=True, savefile="calc_1D_saved.pkl",
                               KLD_thresh=0.1):
        """
        Reweights 1D profile to different N* umbrellas, compares second derivatives
        of biased profiles to kappa, reports the the KL divergences between reweighted profiles and biased profiles,
        and checks that these are under a specific threshold.
        """
        saveloc = self.calcoutdir + "/" + savefile

        if not saved:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        calc = pickle.load(open(saveloc, "rb"))

        betaF_il, _ = WHAM.statistics.win_betaF(Ntw_win, bin_points, umbrella_win, beta,
                                                bin_style='center')
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, umbrella_win,
                                                                         beta, bin_style='center')

        # Check if windows path exists
        if not os.path.exists(self.plotoutdir + "/windows"):
            os.makedirs(self.plotoutdir + "/windows")

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

            plt.savefig(self.plotoutdir + "/windows/" + "binless_reweight_win_log_likelihood_{}.png".format(n_star_win[i]), bbox_inches='tight')
            plt.close()

        # KL divergence check
        D_KL_i = WHAM.statistics.binless_KLD_reweighted_win_betaF(calc, Ntw_win, bin_points,
                                                                  umbrella_win, beta, bin_style='center')
        logger.info("KL divergences:", D_KL_i)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(n_star_win, D_KL_i, 's')
        ax.axhline(y=KLD_thresh)
        ax.set_ylim([0, None])

        ax.set_xlabel(r"$N^*$")
        ax.set_ylabel(r"$D_{KL}$")

        plt.savefig(self.plotoutdir + "/" + "binless_reweight_KLD_log_likelihood.png", bbox_inches='tight')
        plt.close()

        problem_i_vals = np.argwhere(D_KL_i > KLD_thresh).flatten()

        if len(problem_i_vals) > 0:
            logger.warning("Problem N* vals:", np.array(n_star_win)[problem_i_vals])

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

        for i in problem_i_vals:
            betaF_shift = np.min(betaF_il[i])
            ax.plot(bin_points, betaF_il[i] - betaF_shift, 'x--', label=r"$N^*$ = {}".format(n_star_win[i]), color="C{}".format(i))
            ax.plot(bin_points, betaF_il_reweight[i] - betaF_shift, color="C{}".format(i))

        ax.legend()
        ax.set_xlim([0, self.NMAX])
        ax.set_ylim([0, 8])

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$\beta F_{bias, i}$")

        plt.savefig(self.plotoutdir + "/" + "binless_reweight_win_log_likelihood_problem.png", bbox_inches='tight')
        plt.close()

    ############################################################################
    # 1D phi-ensemble reweighting
    ############################################################################

    def run_phi_ensemble_reweight(self, saved=True, savefile="calc_1D_saved.pkl"):
        """
        Reweights 1D profile and calculates average N~ and Var(N~) in the phi-ensemble.
        Uses averages to estimate phi_1_star.
        """
        saveloc = self.calcoutdir + "/" + savefile

        if not saved:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()

        calc = pickle.load(open(saveloc, "rb"))

        phi_vals = np.linspace(self.PHI_BIN_MIN, self.PHI_BIN_MAX, self.PHIBINS)

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

        for i, peak in enumerate(sorted(peaks)):
            logger.debug(beta * phi_vals[peak])
            ax[1].text(beta * phi_vals[peak], N_var_vals[peak], r"$\beta \phi_{}* = {:.3f}$".format(i + 1, beta * phi_vals[peak]))

        plt.savefig(self.plotoutdir + "/binless_log_likelihood_phi_ensemble.png", bbox_inches='tight')
        plt.close()

        # Write to text file
        of = open(self.calcoutdir + "/phi_ensemble.dat", "w")
        of.write("# phi    <N>    <dN^2>\n")
        for i in range(len(phi_vals)):
            of.write("{:.5f} {:.5f} {:.5f}\n".format(phi_vals[i], N_avg_vals[i], N_var_vals[i]))
        of.close()

        # Write peak information to text file
        of = open(self.calcoutdir + "/phi_ensemble_peaks.dat", "w")
        of.write("# phi    beta*phi\n")
        for i, peak in enumerate(sorted(peaks)):
            of.write("{:.5f} {:.5f}\n".format(phi_vals[peak], beta * phi_vals[peak]))
        of.close()

    ############################################################################
    # phi_1* reweighting
    ############################################################################

    def run_reweight_phi_1_star(self, saved=True, savefile="calc_1D_saved.pkl"):
        """
        Reweights 1D profile to phi_1* ensemble.
        """
        saveloc = self.calcoutdir + "/" + savefile

        if not saved:
            self.run_binless_log_likelihood()

        n_star_win, Ntw_win, bin_points, umbrella_win, beta = self.get_test_data()
        calc = pickle.load(open(saveloc, "rb"))

        phi_1_star = self.PHI_STAR
        logger.debug(phi_1_star)

        umb_phi_1_star = potentials.linear(phi_1_star)
        betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, [umb_phi_1_star],
                                                                         beta, bin_style='center')
        betaF_rew = betaF_il_reweight[0]
        betaF_rew = betaF_rew - np.min(betaF_rew)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        indices = np.where(np.logical_and(bin_points >= self.PLOT_PHI_STAR_N_MIN,
                                          bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
        ax.plot(bin_points[indices], betaF_rew[indices], label=r"Biased free energy profile in $\phi_1^*$ ensemble.")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$\beta F$")
        ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX])

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_1_star))

        plt.savefig(self.plotoutdir + "/free_energy_phi_1_star.png", bbox_inches='tight')
        plt.close()

        # Write to text file
        of = open(self.calcoutdir + "/binless_ll_phi_1_star.dat", "w")
        of.write("# N    betaF\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], betaF_rew[i]))
        of.close()

        """Probabilities"""
        delta_x_bin = bin_points[1] - bin_points[0]
        p_bin = delta_x_bin * np.exp(-betaF_rew)

        p_bin = p_bin / (delta_x_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + "binless_ll_phi_1_star_prob.dat", "w")
        of.write("# Nt    Pv(Nt)\n")
        for i in range(len(bin_points)):
            of.write("{:.5f} {:.5f}\n".format(bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(bin_points, p_bin, label="Log-likelihood probability density")
        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$P_v(\tilde{N})$")

        ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])

        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + "binless_ll_phi_1_star_prob.png", bbox_inches='tight')
        plt.close()

    ############################################################################
    # 2D WHAM plot and Rg plot
    ############################################################################

    def run_2D_binless_log_likelihood(self, saved1D=True, saved1Dfile="calc_1D_saved.pkl", saved2D=False, saved2Dfile="calc_2D_saved.pkl"):
        """
        Runs 2D binless log likelihood calculation if 1D data is not available. If 1D data is available, uses 1D point
        weights to re-bin data to 2D.
        """
        saved1Dloc = self.calcoutdir + "/" + saved1Dfile
        saved2Dloc = self.calcoutdir + "/" + saved2Dfile

        n_star_win, Ntw_win, Rg_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

        assert(len(Ntw_win[0]) == len(Rg_win[0]))
        assert(len(Ntw_win[1]) == len(Rg_win[1]))

        # Unroll Ntw_win into a single array
        x_l = Ntw_win[0]
        for i in range(1, len(Ntw_win)):
            x_l = np.hstack((x_l, Ntw_win[i]))

        # Unroll Rg_win into a single array
        y_l = Rg_win[0]
        for i in range(1, len(Rg_win)):
            y_l = np.hstack((y_l, Rg_win[i]))

        N_i = np.array([len(arr) for arr in Ntw_win])

        if saved1D:
            calc = pickle.load(open(saved1Dloc, "rb"))
        elif saved2D:
            calc = pickle.load(open(saved2Dloc, "rb"))
        else:
            # Perform WHAM calculations
            calc = WHAM.binless.Calc1D()
            status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                                solver='log-likelihood',
                                                logevery=1)
            pickle.dump(calc, open(self.calcoutdir + "/calc_2D_saved.pkl", "wb"))

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

        levels = np.linspace(0, self.PLOT_BETAF_MAX, self.PLOT_BETAF_LEVELS)
        cmap = cm.RdYlBu
        contour_filled = ax.contourf(x_bin_points, y_bin_points, betaF_2D_bin.T, levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
        ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(contour_filled, cax=cax, orientation='vertical')

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$R_g$ (nm)")

        cax.set_title(r"$\beta G_v(\tilde{N}, R_g)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.margins(x=0, y=0)

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        plt.savefig(self.plotoutdir + "/binless_2D_log_likelihood.png", bbox_inches='tight')
        plt.close()

        # Write to npy file
        np.save(self.calcoutdir + "/binless_ll_2D.npy", betaF_2D_bin)

        # Write bin points to npy files
        np.save(self.calcoutdir + "/binless_2D_bin_N.npy", x_bin_points)
        np.save(self.calcoutdir + "/binless_2D_bin_Rg.npy", y_bin_points)

        """Probabilities"""
        delta_x_bin = x_bin_points[1] - x_bin_points[0]
        delta_y_bin = y_bin_points[1] - y_bin_points[0]
        p_bin = delta_x_bin * delta_y_bin * np.exp(-betaF_2D_bin)

        p_bin = p_bin / (delta_x_bin * delta_y_bin * np.sum(p_bin))  # normalize

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, np.max(p_bin), self.PLOT_PV_LEVELS)
        cmap = cm.YlGnBu
        contour_filled = ax.contourf(x_bin_points, y_bin_points, p_bin.T, levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
        ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(contour_filled, cax=cax, orientation='vertical', format='%.1e')

        ax.set_xlabel(r"$\tilde{N}$")
        ax.set_ylabel(r"$R_g$ (nm)")

        cax.set_title(r"$P_v(\tilde{N}, R_g)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.margins(x=0, y=0)

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        plt.savefig(self.plotoutdir + "/binless_2D_ll_prob.png", bbox_inches='tight')
        plt.close()

        # Write to npy file
        np.save(self.calcoutdir + "/binless_ll_2D_prob.npy", p_bin)

    def run_2D_bin_Rg(self, saved1D=True, saved1Dfile="calc_1D_saved.pkl", saved2D=False, saved2Dfile="calc_2D_saved.pkl"):
        """
        Calculates 1D profile in Rg by integrating out N coordinate from 2D profile.
        """
        saved1Dloc = self.calcoutdir + "/" + saved1Dfile
        saved2Dloc = self.calcoutdir + "/" + saved2Dfile

        n_star_win, Ntw_win, Rg_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

        assert(len(Ntw_win[0]) == len(Rg_win[0]))
        assert(len(Ntw_win[1]) == len(Rg_win[1]))

        # Unroll Ntw_win into a single array
        x_l = Ntw_win[0]
        for i in range(1, len(Ntw_win)):
            x_l = np.hstack((x_l, Ntw_win[i]))

        # Unroll Rg_win into a single array
        y_l = Rg_win[0]
        for i in range(1, len(Rg_win)):
            y_l = np.hstack((y_l, Rg_win[i]))

        N_i = np.array([len(arr) for arr in Ntw_win])

        if saved1D:
            calc = pickle.load(open(saved1Dloc, "rb"))
        elif saved2D:
            calc = pickle.load(open(saved2Dloc, "rb"))
        else:
            # Perform WHAM calculations
            calc = WHAM.binless.Calc1D()
            status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                                solver='log-likelihood',
                                                logevery=1)
            pickle.dump(calc, open(self.calcoutdir + "/calc_2D_saved.pkl", "wb"))

            # Optimized?
            logger.debug(status)

        g_i = calc.g_i

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        betaF_Rg = calc.bin_second_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                 x_bin_style='center', y_bin_style='center')
        betaF_Rg = betaF_Rg - np.min(betaF_Rg)

        # Write to text file
        of = open(self.calcoutdir + "/" + "binless_ll_Rg.dat", "w")
        of.write("# Nt    betaF\n")
        for i in range(len(y_bin_points)):
            of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], betaF_Rg[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(y_bin_points, betaF_Rg, label="Log-likelihood binless Rg profile")
        ax.set_xlabel(r"$R_g$")
        ax.set_ylabel(r"$\beta F$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + "binless_log_likelihood_Rg.png", bbox_inches='tight')
        plt.close()

        """Probabilities"""
        delta_y_bin = y_bin_points[1] - y_bin_points[0]
        p_bin = delta_y_bin * np.exp(-betaF_Rg)

        p_bin = p_bin / (delta_y_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + "binless_ll_prob_Rg.dat", "w")
        of.write("# Nt    Pv(Rg)\n")
        for i in range(len(y_bin_points)):
            of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(y_bin_points, p_bin, label="Log-likelihood probability density")
        ax.set_xlabel(r"$R_g$")
        ax.set_ylabel(r"$P_v(R_g)$")
        ax.margins(x=0, y=0)

        plt.savefig(self.plotoutdir + "/" + "binless_ll_prob_Rg.png", bbox_inches='tight')
        plt.close()

    ############################################################################
    # 2D phi_1* reweighting
    ############################################################################

    def run_2D_reweight_phi_1_star(self, saved1D=True, saved1Dfile="calc_1D_saved.pkl", saved2D=False, saved2Dfile="calc_2D_saved.pkl"):
        """
        Reweights 2D profile to phi_1* ensemble.
        """
        saved1Dloc = self.calcoutdir + "/" + saved1Dfile
        saved2Dloc = self.calcoutdir + "/" + saved2Dfile

        n_star_win, Ntw_win, Rg_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

        assert(len(Ntw_win[0]) == len(Rg_win[0]))
        assert(len(Ntw_win[1]) == len(Rg_win[1]))

        # Unroll Ntw_win into a single array
        x_l = Ntw_win[0]
        for i in range(1, len(Ntw_win)):
            x_l = np.hstack((x_l, Ntw_win[i]))

        # Unroll Rg_win into a single array
        y_l = Rg_win[0]
        for i in range(1, len(Rg_win)):
            y_l = np.hstack((y_l, Rg_win[i]))

        N_i = np.array([len(arr) for arr in Ntw_win])

        if saved1D:
            calc = pickle.load(open(saved1Dloc, "rb"))
        elif saved2D:
            calc = pickle.load(open(saved2Dloc, "rb"))
        else:
            # Perform WHAM calculations
            calc = WHAM.binless.Calc1D()
            status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                                solver='log-likelihood',
                                                logevery=1)
            pickle.dump(calc, open(self.calcoutdir + "/calc_2D_saved.pkl", "wb"))

            # Optimized?
            logger.debug(status)

        g_i = calc.g_i

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        phi_1_star = self.PHI_STAR2
        logger.debug(phi_1_star)

        G_l_rew = calc.reweight(beta, u_bias=potentials.linear(phi_1_star))

        betaF_2D_rew, (betaF_2D_rew_bin_counts, _1, _2) = calc.bin_2D_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                                                    G_l=G_l_rew, x_bin_style='center', y_bin_style='center')
        betaF_2D_rew = betaF_2D_rew - np.min(betaF_2D_rew)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, self.PLOT_PHI_STAR_BETAF_MAX, self.PLOT_PHI_STAR_BETAF_LEVELS)
        cmap = cm.RdYlBu

        x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN, x_bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)

        y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_RG_MIN, y_bin_points <= self.PLOT_PHI_STAR_RG_MAX))[0]
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
        ax.set_ylabel(r"$R_g$ (nm)")

        cax.set_title(r"$\beta G_v^{\phi_1^*}(\tilde{N}, R_g)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.margins(x=0, y=0)

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_1_star))

        plt.savefig(self.plotoutdir + "/binless_2D_phi_1_star.png", bbox_inches='tight')
        plt.close()

        # Write free energy to npy file
        np.save(self.calcoutdir + "/binless_2D_phi_1_star.npy", betaF_2D_rew)

        # Write bin points to npy files
        np.save(self.calcoutdir + "/binless_2D_phi_1_star_bin_N.npy", x_bin_points)
        np.save(self.calcoutdir + "/binless_2D_phi_1_star_bin_Rg.npy", y_bin_points)

        # Write bin counts to npy file
        np.save(self.calcoutdir + "/binless_2D_phi_1_star_bin_counts.npy", betaF_2D_rew_bin_counts)

        """Probabilities"""
        delta_x_bin = x_bin_points[1] - x_bin_points[0]
        delta_y_bin = y_bin_points[1] - y_bin_points[0]
        p_bin = delta_x_bin * delta_y_bin * np.exp(-betaF_2D_rew)

        p_bin = p_bin / (delta_x_bin * delta_y_bin * np.sum(p_bin))  # normalize

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        levels = np.linspace(0, np.max(p_bin), self.PLOT_PHI_STAR_PV_LEVELS)
        cmap = cm.YlGnBu

        x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN, x_bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)

        y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_RG_MIN, y_bin_points <= self.PLOT_PHI_STAR_RG_MAX))[0]
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
        ax.set_ylabel(r"$R_g$ (nm)")

        cax.set_title(r"$P_v^{\phi_1^*}(\tilde{N}, R_g)$")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.margins(x=0, y=0)

        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_1_star))

        plt.savefig(self.plotoutdir + "/binless_2D_phi_1_star_prob.png", bbox_inches='tight')
        plt.close()

        # Write probabilities to npy file
        np.save(self.calcoutdir + "/binless_2D_phi_1_star_prob.npy", p_bin)

    def run_2D_reweight_phi_1_star_bin_Rg(self, saved1D=True, saved1Dfile="calc_1D_saved.pkl", saved2D=False, saved2Dfile="calc_2D_saved.pkl"):
        """
        Calculates 1D profile in Rg at phi_1_star by integrating out N coordinate from 2D reweighted profile.
        """
        saved1Dloc = self.calcoutdir + "/" + saved1Dfile
        saved2Dloc = self.calcoutdir + "/" + saved2Dfile

        n_star_win, Ntw_win, Rg_win, x_bin_points, y_bin_points, umbrella_win, beta = self.get_test_data2()

        assert(len(Ntw_win[0]) == len(Rg_win[0]))
        assert(len(Ntw_win[1]) == len(Rg_win[1]))

        # Unroll Ntw_win into a single array
        x_l = Ntw_win[0]
        for i in range(1, len(Ntw_win)):
            x_l = np.hstack((x_l, Ntw_win[i]))

        # Unroll Rg_win into a single array
        y_l = Rg_win[0]
        for i in range(1, len(Rg_win)):
            y_l = np.hstack((y_l, Rg_win[i]))

        N_i = np.array([len(arr) for arr in Ntw_win])

        if saved1D:
            calc = pickle.load(open(saved1Dloc, "rb"))
        elif saved2D:
            calc = pickle.load(open(saved2Dloc, "rb"))
        else:
            # Perform WHAM calculations
            calc = WHAM.binless.Calc1D()
            status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                                solver='log-likelihood',
                                                logevery=1)
            pickle.dump(calc, open(self.calcoutdir + "/calc_2D_saved.pkl", "wb"))

            # Optimized?
            logger.debug(status)

        g_i = calc.g_i

        # Useful for debugging:
        logger.debug("Window free energies: ", g_i)

        phi_1_star = self.PHI_STAR2
        logger.debug(phi_1_star)

        G_l_rew = calc.reweight(beta, u_bias=potentials.linear(phi_1_star))

        betaF_Rg_rew = calc.bin_second_betaF_profile(y_l, x_bin_points, y_bin_points,
                                                     G_l=G_l_rew, x_bin_style='center', y_bin_style='center')
        betaF_Rg_rew = betaF_Rg_rew - np.min(betaF_Rg_rew)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_RG_MIN,
                                          y_bin_points <= self.PLOT_PHI_STAR_RG_MAX))[0]
        ax.plot(y_bin_points[indices], betaF_Rg_rew[indices], label=r"Biased free energy profile in $\phi_1^*$ ensemble.")
        ax.set_xlabel(r"$R_g$")
        ax.set_ylabel(r"$\beta F$")
        ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX])

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_1_star))

        plt.savefig(self.plotoutdir + "/free_energy_phi_1_star_Rg.png", bbox_inches='tight')
        plt.close()

        # Write to text file
        of = open(self.calcoutdir + "/binless_ll_phi_1_star_Rg.dat", "w")
        of.write("# N    betaF\n")
        for i in range(len(y_bin_points)):
            of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], betaF_Rg_rew[i]))
        of.close()

        """Probabilities"""
        delta_y_bin = y_bin_points[1] - y_bin_points[0]
        p_bin = delta_y_bin * np.exp(-betaF_Rg_rew)

        p_bin = p_bin / (delta_y_bin * np.sum(p_bin))  # normalize

        # Write to text file
        of = open(self.calcoutdir + "/" + "binless_ll_phi_1_star_prob_Rg.dat", "w")
        of.write("# Nt    Pv(Rg)\n")
        for i in range(len(y_bin_points)):
            of.write("{:.5f} {:.5f}\n".format(y_bin_points[i], p_bin[i]))
        of.close()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(y_bin_points, p_bin, label="Log-likelihood probability density")
        ax.set_xlabel(r"$R_g$")
        ax.set_ylabel(r"$P_v(R_g)$")

        ax.margins(x=0, y=0)

        ax.set_title(r"$\beta \phi = \beta \phi^*$={:.3f}".format(beta * phi_1_star))

        plt.savefig(self.plotoutdir + "/" + "binless_ll_phi_1_star_prob_Rg.png", bbox_inches='tight')
        plt.close()

    ############################################################################
    # phi_c* coexistence integrations
    ############################################################################

    def coexistence_integrations(self, int_Nt=True, int_Rg=True, int_NtRg=True):
        """
        Integrates reweighted 1D profiles in N and Rg, and 2D profile, all att phi_1_star, to determine coexistence.
        """

        if int_Nt:
            """Integrate N"""
            f = open(self.calcoutdir + "/" + "binless_ll_phi_1_star_prob.dat")

            nt = []
            pvnt = []

            for line in f:
                if line.strip().split()[0] != '#':
                    nt.append(float(line.strip().split()[0]))
                    pvnt.append(float(line.strip().split()[1]))

            f.close()

            nt = np.array(nt)
            pvnt = np.array(pvnt)

            idx1 = np.argwhere(nt < self.Nt_split)
            idx2 = np.argwhere(nt >= self.Nt_split)

            dx = nt[1] - nt[0]

            p1 = dx * np.sum(pvnt[idx1].flatten())
            p2 = dx * np.sum(pvnt[idx2].flatten())

            logger.info("Probabilities in N: {:.5f} {:.5f}".format(p1, p2))

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.plot(nt, pvnt, label="Log-likelihood probability density")
            ax.axvline(x=self.Nt_split, label=r"$\tilde{{N}}={:.2f}$".format(self.Nt_split))
            ax.text(0.2, 0.5, "P = {:.2f}".format(p1), transform=ax.transAxes)
            ax.text(0.8, 0.5, "P = {:.2f}".format(p2), transform=ax.transAxes)
            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$P_v(\tilde{N})$")
            ax.legend()
            ax.margins(x=0, y=0)
            ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])

            plt.savefig(self.plotoutdir + "/" + "coex_Nt.png", bbox_inches='tight')
            plt.close()

            # Plot free energies
            f = open(self.calcoutdir + "/" + "binless_ll_phi_1_star.dat")

            fnt = []

            for line in f:
                if line.strip().split()[0] != '#':
                    fnt.append(float(line.strip().split()[1]))

            fnt = np.array(fnt)

            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.plot(nt, fnt, label="Log-likelihood free energy")
            ax.axvline(x=self.Nt_split, label=r"$\tilde{{N}}={:.2f}$".format(self.Nt_split))
            ax.text(0.2, 0.5, "P = {:.2f}".format(p1), transform=ax.transAxes)
            ax.text(0.8, 0.5, "P = {:.2f}".format(p2), transform=ax.transAxes)
            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$F(\tilde{N})$")
            ax.legend()
            ax.margins(x=0, y=0)
            ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])
            ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX])

            plt.savefig(self.plotoutdir + "/" + "coex_Nt_free_energy.png", bbox_inches='tight')
            plt.close()

        if int_Rg:
            """Integrate Rg"""
            f = open(self.calcoutdir + "/" + "binless_ll_phi_1_star_prob_Rg.dat")

            rg = []
            pvrg = []

            for line in f:
                if line.strip().split()[0] != '#':
                    rg.append(float(line.strip().split()[0]))
                    pvrg.append(float(line.strip().split()[1]))

            f.close()

            rg = np.array(rg)
            pvrg = np.array(pvrg)

            idx1 = np.argwhere(rg < self.Rg_split)
            idx2 = np.argwhere(rg >= self.Rg_split)

            dx = rg[1] - rg[0]

            p1 = dx * np.sum(pvrg[idx1].flatten())
            p2 = dx * np.sum(pvrg[idx2].flatten())

            logger.info("Probabilities in Rg: {:.5f} {:.5f}".format(p1, p2))

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.plot(rg, pvrg, label="Probability density")
            ax.axvline(x=self.Rg_split, label=r"$R_g ={:.2f}$".format(self.Rg_split))
            ax.text(0.2, 0.5, "P = {:.2f}".format(p1), transform=ax.transAxes)
            ax.text(0.8, 0.5, "P = {:.2f}".format(p2), transform=ax.transAxes)
            ax.set_xlabel(r"$R_g$")
            ax.set_ylabel(r"$P_v(R_g)$")
            ax.legend()
            ax.margins(x=0, y=0)

            ax.set_xlim([self.PLOT_PHI_STAR_RG_MIN, self.PLOT_PHI_STAR_RG_MAX])

            plt.savefig(self.plotoutdir + "/" + "coex_Rg.png", bbox_inches='tight')
            plt.close()

            # Plot free energies
            f = open(self.calcoutdir + "/" + "binless_ll_phi_1_star_Rg.dat")

            frg = []

            for line in f:
                if line.strip().split()[0] != '#':
                    frg.append(float(line.strip().split()[1]))

            f.close()

            frg = np.array(frg)

            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.plot(rg, frg, label="Free energy")
            ax.axvline(x=self.Rg_split, label=r"$R_g ={:.2f}$".format(self.Rg_split))
            ax.text(0.2, 0.5, "P = {:.2f}".format(p1), transform=ax.transAxes)
            ax.text(0.8, 0.5, "P = {:.2f}".format(p2), transform=ax.transAxes)
            ax.set_xlabel(r"$R_g$")
            ax.set_ylabel(r"$F(R_g)$")
            ax.legend()
            ax.margins(x=0, y=0)
            ax.set_xlim([self.PLOT_PHI_STAR_RG_MIN, self.PLOT_PHI_STAR_RG_MAX])
            ax.set_ylim([0, self.PLOT_PHI_STAR_BETAF_MAX])

            plt.savefig(self.plotoutdir + "/" + "coex_Rg_free_energy.png", bbox_inches='tight')
            plt.close()

        if int_NtRg:
            """Integrate N, Rg"""
            pvntrg = np.load(self.calcoutdir + "/binless_2D_phi_1_star_prob.npy")
            x_bin_points = np.load(self.calcoutdir + "/binless_2D_phi_1_star_bin_N.npy")
            y_bin_points = np.load(self.calcoutdir + "/binless_2D_phi_1_star_bin_Rg.npy")

            xv, yv = np.meshgrid(x_bin_points, y_bin_points, indexing='ij')
            logger.debug(xv.shape)
            mask1 = yv < self.NtRg_split_m * (xv - self.NtRg_split_x0) + self.NtRg_split_c
            mask2 = yv >= self.NtRg_split_m * (xv - self.NtRg_split_x0) + self.NtRg_split_c

            dx = x_bin_points[1] - x_bin_points[0]
            dy = y_bin_points[1] - y_bin_points[0]

            p1 = dx * dy * np.sum(mask1 * pvntrg)
            p2 = dx * dy * np.sum(mask2 * pvntrg)

            logger.info("Probabilities in N, Rg: {:.5f} {:.5f}".format(p1, p2))

            # Plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

            levels = np.linspace(0, np.max(pvntrg), self.PLOT_PHI_STAR_PV_LEVELS)
            cmap = cm.YlGnBu

            x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN, x_bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)

            y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_RG_MIN, y_bin_points <= self.PLOT_PHI_STAR_RG_MAX))[0]
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            contour_filled = ax.contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                         pvntrg[x_min:x_max, y_min:y_max].T,
                                         levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
            ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            fig.colorbar(contour_filled, cax=cax, orientation='vertical', format='%.1e')

            # Plot dividing line
            ax.plot(x_bin_points[x_indices], self.NtRg_split_m * (x_bin_points[x_indices] - self.NtRg_split_x0) + self.NtRg_split_c)
            ax.plot(self.Nt_split, self.Rg_split, 'x')
            ax.text(self.Nt_split + 5, self.Rg_split + .5, "({:.2f}, {:.2f})".format(self.Nt_split, self.Rg_split))

            ax.text(0.2, 0.2, "P = {:.2f}".format(p1), transform=ax.transAxes)
            ax.text(0.8, 0.8, "P = {:.2f}".format(p2), transform=ax.transAxes)

            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$R_g$ (nm)")

            cax.set_title(r"$P_v^{\phi_1^*}(\tilde{N}, R_g)$")

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])
            ax.set_ylim([self.PLOT_PHI_STAR_RG_MIN, self.PLOT_PHI_STAR_RG_MAX])

            ax.tick_params(axis='both', which='both', direction='in', pad=10)

            plt.savefig(self.plotoutdir + "/" + "coex_NtRg.png", bbox_inches='tight')
            plt.close()

            # Plot free energies
            fntrg = np.load(self.calcoutdir + "/binless_2D_phi_1_star.npy")

            fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

            levels = np.linspace(0, self.PLOT_PHI_STAR_BETAF_MAX, self.PLOT_PHI_STAR_BETAF_LEVELS)
            cmap = cm.RdYlBu

            x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN, x_bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)

            y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_RG_MIN, y_bin_points <= self.PLOT_PHI_STAR_RG_MAX))[0]
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            contour_filled = ax.contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                         fntrg[x_min:x_max, y_min:y_max].T,
                                         levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
            ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            fig.colorbar(contour_filled, cax=cax, orientation='vertical')

            # Plot dividing line
            ax.plot(x_bin_points[x_indices], self.NtRg_split_m * (x_bin_points[x_indices] - self.NtRg_split_x0) + self.NtRg_split_c)
            ax.plot(self.Nt_split, self.Rg_split, 'x')
            ax.text(self.Nt_split + 5, self.Rg_split + .5, "({:.2f}, {:.2f})".format(self.Nt_split, self.Rg_split))

            ax.text(0.2, 0.2, "P = {:.2f}".format(p1), transform=ax.transAxes)
            ax.text(0.8, 0.8, "P = {:.2f}".format(p2), transform=ax.transAxes)

            ax.set_xlabel(r"$\tilde{N}$")
            ax.set_ylabel(r"$R_g$ (nm)")

            cax.set_title(r"$\beta G_v^{\phi_1^*}(\tilde{N}, R_g)$")

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])
            ax.set_ylim([self.PLOT_PHI_STAR_RG_MIN, self.PLOT_PHI_STAR_RG_MAX])

            ax.tick_params(axis='both', which='both', direction='in', pad=10)

            plt.savefig(self.plotoutdir + "/" + "coex_NtRg_free_energy.png", bbox_inches='tight')
            plt.close()

            """Plot masks"""
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=600)

            levels = np.linspace(0, np.max(pvntrg), self.PLOT_PHI_STAR_PV_LEVELS)
            cmap = cm.YlGnBu

            x_indices = np.where(np.logical_and(x_bin_points >= self.PLOT_PHI_STAR_N_MIN, x_bin_points <= self.PLOT_PHI_STAR_N_MAX))[0]
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)

            y_indices = np.where(np.logical_and(y_bin_points >= self.PLOT_PHI_STAR_RG_MIN, y_bin_points <= self.PLOT_PHI_STAR_RG_MAX))[0]
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            # FIRST MASK
            contour_filled = ax[0].contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                            (mask1 * pvntrg)[x_min:x_max, y_min:y_max].T,
                                            levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
            ax[0].contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)

            # Plot dividing line
            ax[0].plot(x_bin_points[x_indices], self.NtRg_split_m * (x_bin_points[x_indices] - self.NtRg_split_x0) + self.NtRg_split_c)
            ax[0].plot(self.Nt_split, self.Rg_split, 'x')
            ax[0].text(self.Nt_split + 5, self.Rg_split + .5, "({:.2f}, {:.2f})".format(self.Nt_split, self.Rg_split))

            ax[0].text(0.2, 0.2, "P = {:.2f}".format(p1), transform=ax[0].transAxes)

            ax[0].set_xlabel(r"$\tilde{N}$")
            ax[0].set_ylabel(r"$R_g$ (nm)")

            ax[0].xaxis.set_minor_locator(AutoMinorLocator())
            ax[0].yaxis.set_minor_locator(AutoMinorLocator())

            ax[0].set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])
            ax[0].set_ylim([self.PLOT_PHI_STAR_RG_MIN, self.PLOT_PHI_STAR_RG_MAX])

            ax[0].tick_params(axis='both', which='both', direction='in', pad=10)

            # SECOND MASK
            contour_filled = ax[1].contourf(x_bin_points[x_min:x_max], y_bin_points[y_min:y_max],
                                            (mask2 * pvntrg)[x_min:x_max, y_min:y_max].T,
                                            levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
            ax[1].contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)

            # Plot dividing line
            ax[1].plot(x_bin_points[x_indices], self.NtRg_split_m * (x_bin_points[x_indices] - self.NtRg_split_x0) + self.NtRg_split_c)
            ax[1].plot(self.Nt_split, self.Rg_split, 'x')
            ax[1].text(self.Nt_split + 5, self.Rg_split + .5, "({:.2f}, {:.2f})".format(self.Nt_split, self.Rg_split))

            ax[1].text(0.8, 0.8, "P = {:.2f}".format(p2), transform=ax[1].transAxes)

            ax[1].set_xlabel(r"$\tilde{N}$")
            ax[1].set_ylabel(r"$R_g$ (nm)")

            ax[1].xaxis.set_minor_locator(AutoMinorLocator())
            ax[1].yaxis.set_minor_locator(AutoMinorLocator())

            ax[1].set_xlim([self.PLOT_PHI_STAR_N_MIN, self.PLOT_PHI_STAR_N_MAX])
            ax[1].set_ylim([self.PLOT_PHI_STAR_RG_MIN, self.PLOT_PHI_STAR_RG_MAX])

            ax[1].tick_params(axis='both', which='both', direction='in', pad=10)

            plt.savefig(self.plotoutdir + "/" + "coex_NtRg_int_regions.png", bbox_inches='tight')
            plt.close()

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

        phi_vals = np.linspace(self.PHI_BIN_MIN, self.PHI_BIN_MAX, self.PHIBINS)

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

    def run_bootstrap_ll_phi_ensemble(self, nboot=10, nworkers=8):
        """
        Runs 1D binless log likelihood calculation and phi-ensemble reweighting.
        """
        with Pool(processes=nworkers) as pool:
            ret_dicts = pool.map(self.boot_worker, range(nboot))

        # Unpack returned values, calculate error bars, etc
        betaF_all = []
        N_avg_all = []
        N_var_all = []
        for boot_idx in range(nboot):
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
        of = open(self.calcoutdir + "/" + "binless_ll_bootstrap.dat", "w")
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

        plt.savefig(self.plotoutdir + "/" + "binless_log_likelihood_bootstrap.png", bbox_inches='tight')
        plt.close()

        N_avg = N_avg_all.mean(axis=0)
        N_avg_err = N_avg_all.std(axis=0)

        N_var = N_var_all.mean(axis=0)
        N_var_err = N_var_all.std(axis=0)

        phi_vals = np.linspace(self.PHI_BIN_MIN, self.PHI_BIN_MAX, self.PHIBINS)

        # Write to text file
        of = open(self.calcoutdir + "/" + "phi_ensemble_bootstrap.dat", "w")
        of.write("# phi    <N>    sem(N)   <dN^2>    sem(dN^2)\n")
        for i in range(len(phi_vals)):
            of.write("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(phi_vals[i], N_avg[i], N_avg_err[i],
                                                                   N_var[i], N_var_err[i]))
        of.close()

        phi_1_stars = np.zeros(nboot)
        phi_2_stars = np.zeros(nboot)

        # START
        all_peaks = []

        # Calculate phi_1_star and error bars on phi_1_star from bootstrapping
        for nb in range(nboot):
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

        of = open(self.calcoutdir + "/" + "phi_peaks_bootstrap.dat", "w")

        of.write("Peak phi values:\n")
        for phi_peaks in all_peaks:
            of.write(" ".join(["{:.5f}".format(peak) for peak in beta * phi_peaks]) + "\n")
        of.write("\n")

        of.write("beta phi_1* and beta phi_2* values:\n")
        for nb in range(nboot):
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

        plt.savefig(self.plotoutdir + "/binless_log_likelihood_phi_ensemble_bootstrap.png", bbox_inches='tight')
        plt.close()

        of.close()

    ############################################################################
    # deltaGu calculation by basin differences
    ############################################################################

    def calc_deltaGu_diff_method(self, boot_errors=True):
        """Calculate difference between N_C and N_E points to get deltaGu.
        If saved bootstrap errors are available, use these to get error bars on deltaGu."""
        f = open(self.calcoutdir + "/" + "binless_ll.dat")

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

        if boot_errors:
            f = open(self.calcoutdir + "/" + "binless_ll_bootstrap.dat")

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

        return deltaGu, deltaGuerr

    ############################################################################
    # deltaGu calculation by basin integrations
    ############################################################################

    def calc_deltaGu_int_method_1D(self):
        """Calculate ratio of C basin to E basin probabilities, in N~, to get deltaGu."""
        f = open(self.calcoutdir + "/" + "binless_ll.dat")

        nt = []
        fnt = []

        for line in f:
            if line.strip().split()[0] != '#':
                nt.append(float(line.strip().split()[0]))
                fnt.append(float(line.strip().split()[1]))

        f.close()

        nt = np.array(nt)
        fnt = np.array(fnt)

        idx1 = np.argwhere(nt < self.Nt_split)
        idx2 = np.argwhere(nt >= self.Nt_split)

        g1 = -logsumexp(-fnt[idx1].flatten())
        g2 = -logsumexp(-fnt[idx2].flatten())

        return(g2 - g1)

    def calc_deltaGu_int_method_2D(self):
        """Calculate ratio of C basin to E basin probabilities, in 2D, to get deltaGu."""
        pass

    ############################################################################
    # computation call
    ############################################################################

    def __call__(self, calc_types, calc_args={}):
        for calc_type in calc_types:
            if calc_type == "all":
                self.plot_hist()
                self.run_binless_log_likelihood()
                self.run_kappa_checks()
                self.run_reweighting_checks()
                self.run_phi_ensemble_reweight()
                self.run_reweight_phi_1_star()
                self.run_2D_binless_log_likelihood()
                self.run_2D_reweight_phi_1_star()
                self.run_2D_bin_Rg()
                self.run_2D_reweight_phi_1_star_bin_Rg()
                self.coexistence_integrations()
                self.run_bootstrap_ll_phi_ensemble(calc_args.get('nboot', 10), calc_args.get('nworkers', 8))
            elif calc_type == "hist":
                self.plot_hist()
            elif calc_type == "1D":
                self.run_binless_log_likelihood()
            elif calc_type == "kappa":
                self.run_kappa_checks()
            elif calc_type == "KLD":
                self.run_reweighting_checks()
            elif calc_type == "1D_phi":
                self.run_phi_ensemble_reweight()
            elif calc_type == "1D_phi_1_star":
                self.run_reweight_phi_1_star()
            elif calc_type == "2D":
                self.run_2D_binless_log_likelihood()
            elif calc_type == "2D_phi_1_star":
                self.run_2D_reweight_phi_1_star()
            elif calc_type == "Rg":
                self.run_2D_bin_Rg()
            elif calc_type == "Rg_phi_1_star":
                self.run_2D_reweight_phi_1_star_bin_Rg()
            elif calc_type == "coex":
                self.coexistence_integrations()
            elif calc_type == "coex-Nt":
                self.coexistence_integrations(int_Nt=True, int_Rg=False, int_NtRg=False)
            elif calc_type == "coex-Rg":
                self.coexistence_integrations(int_Nt=False, int_Rg=True, int_NtRg=False)
            elif calc_type == "coex-NtRg":
                self.coexistence_integrations(int_Nt=False, int_Rg=False, int_NtRg=True)
            elif calc_type == "1D_boot":
                self.run_bootstrap_ll_phi_ensemble(calc_args.get('nboot', 10), calc_args.get('nworkers', 8))
            elif calc_type == "deltaG_diff":
                deltaGu, deltaGuerr = self.calc_deltaGu_diff_method(boot_errors=True)
                print(deltaGu, deltaGuerr)
            elif calc_type == "deltaG_diff_noerr":
                deltaGu, _ = self.calc_deltaGu_diff_method(boot_errors=False)
                print(deltaGu)
            elif calc_type == "deltaG_int_1D":
                deltaGu = self.calc_deltaGu_int_method_1D()
                print(deltaGu)
            else:
                raise ValueError("Calculation type not recognized.")

def main():
    pass


if __name__ == "__main__":
    main()
