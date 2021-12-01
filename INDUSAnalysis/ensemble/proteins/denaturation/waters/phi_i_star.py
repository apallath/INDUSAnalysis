"""
Plots ni v/s phi and phi_i* for a set of representative atoms, and also for each atom i.

Stores calculated phi_i* values.
"""
import argparse
from functools import partial
import os
import warnings
import pickle


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import MDAnalysis as mda
import numpy as np
import scipy
from tqdm import tqdm

from INDUSAnalysis.lib.collective import phi_to_P, P_to_phi, \
    linear_model, fit_linear_model, \
    integrated_step_gaussian, derivative_integrated_step_gaussian, fit_integrated_step_gaussian
from INDUSAnalysis.timeseries import TimeSeries, TimeSeriesAnalysis


def phi_i_star(phivals: list,
               runs: list,
               start_time: int,
               structfile: str,
               calc_dir: str,
               ni_format: str,
               sample_imgfile: str,
               all_imgformat: str,
               pklfile: str,
               buried_cut: float,
               phi_star_collective: float,
               plot_probe_indices: list,
               P0=1,
               D_by_A_guess=5,
               E_guess=0.05,
               F_alpha_level=0.001):

    tsa = TimeSeriesAnalysis()

    ############################################################################
    # Load data
    ############################################################################

    u = mda.Universe(structfile)
    protein_heavy = u.select_atoms("protein and not name H*")
    protein_heavy_indices = protein_heavy.atoms.indices

    meanwaters = np.zeros((len(phivals), len(runs), len(protein_heavy_indices)))
    varwaters = np.zeros((len(phivals), len(runs), len(protein_heavy_indices)))

    # All other phi values
    for idx, phi in enumerate(phivals):
        for runidx, run in enumerate(runs):
            ts = tsa.load_TimeSeries(calc_dir + ni_format.format(phi=phi, run=run))
            ts = ts[start_time:]
            run_waters = ts.data_array[:, protein_heavy_indices]

            # Calcuate per-atom mean waters and var waters for each run
            mean_run_waters = np.mean(run_waters, axis=0)
            var_run_waters = np.var(run_waters, axis=0)

            # Append per-atom mean and var waters for each run
            meanwaters[idx, runidx, :] = mean_run_waters
            varwaters[idx, runidx, :] = var_run_waters

    mean_meanwaters = np.mean(meanwaters, axis=1)
    std_meanwaters = np.std(meanwaters, axis=1)

    phivals = np.array([float(phi) for phi in phivals])
    order = np.argsort(phivals)

    ############################################################################
    # Sample fits
    ############################################################################
    plot_probe_indices = [int(x) for x in plot_probe_indices]

    fig, ax = plt.subplots(2, 1, figsize=(12, 16), dpi=300)

    for ixx, probe in enumerate(plot_probe_indices):
        order = np.argsort(phivals)
        xdata = np.array(phivals)[order]
        ydata = mean_meanwaters[:, probe][order]
        yerr = std_meanwaters[:, probe][order]

        # Disallow zero errors
        for idx, val in enumerate(yerr):
            if val < 1e-3:
                yerr[idx] = 1e-3

        # Plot original data
        ax[0].errorbar(xdata, ydata, yerr=yerr,
                       barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none",
                       label="Probe on h. atom " + str(probe), color="C{}".format(ixx))
        ax[0].fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.2)

        # Initial parameter guesses
        A_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        B_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])

        dy_data = ydata[1:] - ydata[0:-1]
        dx_data = xdata[1:] - xdata[0:-1]
        dydx_data = dy_data / dx_data
        C_guess_idx = np.argmin(dydx_data) + 1

        C_guess = xdata[C_guess_idx]
        D_guess = D_by_A_guess * A_guess
        F_guess = ydata[C_guess_idx]

        p_guess = [A_guess, B_guess, C_guess, D_guess, E_guess, F_guess]

        # Fit
        try:
            popt, perr, chi_sq = fit_integrated_step_gaussian(xdata, ydata, yerr, p_guess)

            x_fit_data = np.linspace(min(phivals), max(phivals), 200)
            y_fit_data = integrated_step_gaussian(x_fit_data, *popt)
            ax[0].plot(x_fit_data, y_fit_data, color="C{}".format(ixx))

            # Calculate gradients
            dydx = derivative_integrated_step_gaussian(x_fit_data, *popt)

            ax[1].plot(x_fit_data, -dydx, label="Probe on h. atom " + str(probe), color="C{}".format(ixx))

        except RuntimeError:
            print("Couldn't fit probe {}".format(probe))

    for i in range(2):
        x_minor_locator = AutoMinorLocator(10)
        y_minor_locator = AutoMinorLocator(10)
        ax[i].xaxis.set_minor_locator(x_minor_locator)
        ax[i].yaxis.set_minor_locator(y_minor_locator)
        ax[i].grid(which='major', linestyle='-')
        ax[i].grid(which='minor', linestyle=':')

        phi_to_P_custom = partial(phi_to_P, P0=P0)
        P_to_phi_custom = partial(P_to_phi, P0=P0)

        secax = ax[i].secondary_xaxis('top', functions=(phi_to_P_custom, P_to_phi_custom))
        secax.set_xlabel(r"Effective hydration shell pressure, $P$ (kbar)")
        ax[i].legend()

    ax[0].set_xlabel(r"$\phi$")
    ax[0].set_ylabel(r"Mean probe waters $\langle n_i \rangle$")

    ax[1].set_xlabel(r"$\phi$")
    ax[1].set_ylabel(r"Susceptibility $-d\langle n_i \rangle / d\phi$")

    plt.savefig(sample_imgfile)

    # Stopped here
    # Edit here onwards

    ############################################################################
    # Actual fits

    # Fit all probes to linear and integrated step gaussian models with different
    # initial guesses, compare fit quality, and compute phi_i_star
    ############################################################################

    """
    # Use text-only Matplotlib backend
    matplotlib.use('Agg')

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Debug
    probes = range(len(protein_heavy_indices))

    phi_i_stars = np.zeros(len(protein_heavy_indices))
    phi_i_star_errors = np.zeros(len(protein_heavy_indices))
    delta_ni_trans = np.zeros(len(protein_heavy_indices))
    delta_phi_trans = np.zeros(len(protein_heavy_indices))

    buried_surface_mask = np.zeros(len(protein_heavy_indices))
    buried_mask = mean_meanwaters[0, np.array(probes)] < BURIED_CUTOFF
    np.save("buried_heavy_atoms_mask.npy", buried_mask)

    if not os.path.exists('phi_i_star_images'):
        os.makedirs('phi_i_star_images')

    for probe in tqdm(probes):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        order = np.argsort(phivals)
        xdata = np.array(phivals)[order]
        ydata = mean_meanwaters[:, probe][order]
        yerr = std_meanwaters[:, probe][order]
        # Disallow zero errors
        for idx, val in enumerate(yerr):
            if val < 1e-3:
                yerr[idx] = 1e-3

        # Plot original data
        ax.errorbar(xdata, ydata, yerr=yerr,
                    barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none")
        ax.fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.3)

        # Initial guess through smart parameter estimation
        A_guess_0 = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        B_guess_0 = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])

        dy_data = ydata[1:] - ydata[0:-1]
        dx_data = xdata[1:] - xdata[0:-1]
        dydx_data = dy_data / dx_data
        C_guess_idx = np.argmin(dydx_data) + 1

        C_guess_0 = xdata[C_guess_idx]
        D_guess_0 = 2 * A_guess_0
        E_guess_0 = 0.1
        F_guess_0 = ydata[C_guess_idx]

        p_guess_0 = [A_guess_0, B_guess_0, C_guess_0, D_guess_0, E_guess_0, F_guess_0]

        # Initial guess with inflection at 0
        A_guess_123 = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        B_guess_123 = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        C_guess_1 = 0  # Zero
        D_guess_123 = 2 * (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        E_guess_123 = 0.1
        F_guess_123 = 10
        p_guess_1 = [A_guess_123, B_guess_123, C_guess_1, D_guess_123, E_guess_123, F_guess_123]

        # Initial guess with inflection at global phi*
        C_guess_2 = PHI_STAR_COLLECTIVE  # Global phi*
        p_guess_2 = [A_guess_123, B_guess_123, C_guess_2, D_guess_123, E_guess_123, F_guess_123]

        # Initial guess with inflection at positive phi
        C_guess_3 = -PHI_STAR_COLLECTIVE  # Global phi*
        p_guess_3 = [A_guess_123, B_guess_123, C_guess_3, D_guess_123, E_guess_123, F_guess_123]

        atom = protein_heavy.atoms[probe]
        atom_name = "{}{}:{}".format(atom.resname, atom.resid, atom.name)
        title = r"H. atom {} ({})".format(probe, atom_name)

        popts = []
        perrs = []
        chi_sqs = []

        for p_guess in [p_guess_0, p_guess_1, p_guess_2, p_guess_3]:
            try:
                popt, perr, chi_sq = fit_integrated_step_gaussian(xdata, ydata, yerr, p_guess)
                chi_sqs.append(chi_sq)
                popts.append(popt)
                perrs.append(perr)

            except RuntimeError:
                chi_sqs.append(np.inf)
                popts.append([])
                perrs.append([])

        chi_sqs = np.array(chi_sqs)

        if not (np.all(chi_sqs > 1e6)):
            method = np.argmin(chi_sqs)

            A = popts[method][0]
            B = popts[method][1]
            C = popts[method][2]
            D = popts[method][3]
            E = popts[method][4]
            F = popts[method][5]

            x_span = np.linspace(min(phivals), max(phivals), 100)
            x_transition = np.linspace(C - 3 * E, C + 3 * E, 100)
            x_fit_data = np.sort(np.hstack((x_span, x_transition)))
            y_fit_data = integrated_step_gaussian(x_fit_data, *popts[method])
            dydx = derivative_integrated_step_gaussian(x_fit_data, *popts[method])

            # Reduced chi sq
            # chi_sq_red = chi_sq/(N - n), N = # of data points, n = # of fit parameters
            chi_sq_nonlinear = chi_sqs[method]
            chi_sq_red_nonlinear = chi_sqs[method] / (len(xdata) - 6)

            ax.plot(x_fit_data, y_fit_data, label=r"Nonlinear fit, $\chi^2_{{red}}$ = {:.2f}"
                    .format(chi_sq_red_nonlinear))
            ax.plot(x_fit_data, -dydx, linestyle="--", label="Susceptibility")

            # Difference points
            ax.plot(C - E, integrated_step_gaussian(C - E, *popts[method]), 'x', color="black", label=r"1 std ($\sigma$) markers")
            ax.plot(C + E, integrated_step_gaussian(C + E, *popts[method]), 'x', color="black")
            ax.plot(C - E, -derivative_integrated_step_gaussian(C - E, *popts[method]), 'x', color="black")
            ax.plot(C + E, -derivative_integrated_step_gaussian(C + E, *popts[method]), 'x', color="black")

            Cerr = perrs[method][2]

            delta_ni = integrated_step_gaussian(C - E, *popts[method]) - integrated_step_gaussian(C + E, *popts[method])
            delta_phi = 2 * E

            ax.text(C, 0.9 * UPPER_WATERS_LIMIT,
                    r"Guess {}, $C$ = {:.2f} $\pm$ {:.2f} kJ/mol".format(method, C, Cerr), bbox=dict(facecolor='C3', alpha=0.5))
            ax.axvspan(C - Cerr, C + Cerr, alpha=0.3, color='C3')

            ax.text(C, 0.8 * UPPER_WATERS_LIMIT, r"$\Delta \langle n_i \rangle = {:.2f}; \Delta \langle n_i \rangle _{{trans}} / \Delta \phi_{{trans}}$ = {:.2f} / {:.2f}"
                    .format(ydata[0] - ydata[-1], delta_ni, delta_phi),
                    bbox=dict(facecolor='C1', alpha=0.5))
            ax.text(0.9 * min(xdata), 0.1 * UPPER_WATERS_LIMIT,
                    r"$A = {:.2f}, D + \frac{{A + B}}{{2}}$ = {:.2f}, B = {:.2f}".format(A, D + (A + B) / 2, B), bbox=dict(facecolor='C2', alpha=0.5))
            ax.axvline(x=C, linestyle=":", color="C2")

        else:
            print("Failed at {}".format(probe))
            chi_sq_red_nonlinear = np.inf
            Cerr = np.inf

        # Compare to linear fit

        # Fit model
        p_guess_lin = [-1, ydata[-1]]
        popt_lin, perr_lin, chi_sq_linear = fit_linear_model(xdata, ydata, yerr, p_guess_lin)
        chi_sq_red_linear = chi_sq_linear / (len(xdata) - 2)

        x_fit_data = np.linspace(min(phivals), max(phivals), 100)
        y_fit_data = linear_model(x_fit_data, *popt_lin)

        ########################################################################
        # Decide whether to use linear or non-linear fit
        #
        # Conditions for linear fit:
        # - phi_i^* has large errors (Error in phi_i* >= 2)
        # - There is no peak => value of -ve derivative at C is less than value at A and B
        # - F-test (https://en.wikipedia.org/wiki/F-test) fails, i.e. F < critical F
        ########################################################################

        # Compute F-statistic
        F = ((chi_sq_linear - chi_sq_nonlinear) / (6 - 2)) / chi_sq_red_nonlinear

        # Degrees of freedom for F-distribution
        dfn = 6 - 2  # numerator degree of freedom for F-statistic
        dfd = (len(xdata) - 6)  # denominator degree of freedom for F-statistic
        # Compute critical value on F-distribution with dfn and dfd degrees of freedom
        crit_F = scipy.stats.f.ppf(q=1 - F_ALPHA_LEVEL, dfn=dfn, dfd=dfd)
        # Reject if F > crit_F

        ax.text(0.9 * min(xdata), 0.2 * UPPER_WATERS_LIMIT,
                r"$F = {:.2f}$; $F_{{crit}}$ @ $\alpha$ = {:.2e} = {:.2f}".format(F, F_ALPHA_LEVEL, crit_F),
                bbox=dict(facecolor='C2', alpha=0.5))

        if Cerr > 2 or (D + (A + B) / 2 > min(A, B)) or (F < crit_F):
            phi_i_star = np.inf
            phi_i_star_error = np.inf
            delta_ni = 0
            delta_phi = 0
            # Draw solid line
            ax.plot(x_fit_data, y_fit_data, label=r"Linear fit, $\chi^2_{{red}}$ = {:.2f}"
                    .format(chi_sq_red_linear), linestyle="-")
        else:
            phi_i_star = C
            phi_i_star_error = Cerr
            # Draw dashed line
            ax.plot(x_fit_data, y_fit_data, label=r"Linear fit, $\chi^2_{{red}}$ = {:.2f}"
                    .format(chi_sq_red_linear), linestyle=":")

        title = title + r", $\phi_i^*$ = {:.2f}".format(phi_i_star)

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True)
        ax.grid()

        ax.set_xlabel(r"$\phi$ (kJ/mol)")
        ax.set_ylabel(r"$\langle n_i \rangle$")

        ax.set_xlim([min(phivals), max(phivals)])
        ax.set_ylim([-0.5, UPPER_WATERS_LIMIT])

        ax.set_title(title)

        secax = ax.secondary_xaxis('top', functions=(phi_to_P, P_to_phi))
        secax.set_xlabel(r"Effective hydration shell pressure, $P$ (kbar)")

        plt.savefig("phi_i_star_images/nonlinear_vs_linear_fit_{}.png".format(probe))
        plt.close()

        # Store data
        phi_i_stars[probe] = phi_i_star
        phi_i_star_errors[probe] = phi_i_star_error
        delta_ni_trans[probe] = delta_ni
        delta_phi_trans[probe] = delta_phi

    # Save phi_i_star data and errors to file
    phi_i_star_data = dict()
    phi_i_star_data['phi_i_stars'] = phi_i_stars
    phi_i_star_data['phi_i_star_errors'] = phi_i_star_errors
    phi_i_star_data['delta_ni_trans'] = delta_ni_trans
    phi_i_star_data['delta_phi_trans'] = delta_phi_trans

    with open("phi_i_star_data.pkl", "wb") as outfile:
        pickle.dump(phi_i_star_data, outfile)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("-Ntw_format", help="format of .pkl file containing Ntw, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-imgfile", help="output image (default=phi_star.png)")
    parser.add_argument("-D_by_A_guess", default=5, help="initial guess for two-state model D/A parameter (default=5)")
    parser.add_argument("-E_guess", default=0.05, help="initial guess for two-state model E parameter (default=0.05)")
    parser.add_argument("-P0", type=float, default=1, help="simulation pressure, in bar (default=1)")

    a = parser.parse_args()

    phi_ensemble(a.phi, a.runs, a.start, a.calc_dir, a.Ntw_format, a.imgfile, a.D_by_A_guess, a.E_guess, a.P0)
