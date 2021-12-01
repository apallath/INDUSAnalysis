"""
Plots unfolding and folding RMSFs of one or two protein structures, given BLAST alignment with STRIDE secondary structures.
"""
import argparse
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

from INDUSAnalysis.protein_order_params import OrderParamsAnalysis as OPA
from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis


matplotlib.use('Agg')


def RMSF(nprot: int,
         structfs: list,
         names: list,
         seq_file: str,
         phivals: list,
         runs: list,
         start_time: int,
         calc_dirs: list,
         di_formats: list,
         imgformat: str,
         plot_native: bool,
         set_max: bool,
         make_movie: bool,
         moviefile: str):

    print("Already generated the images?")
    print("Here's the bash command to stitch them into a movie:")
    print("ffmpeg -r 0.5 -i {} -vcodec mpeg4 -y -vb 200M -q:v 1 {}".format(imgformat.format(phi=r"movie.%05d"), moviefile))

    ############################################################################
    # Read sequence alignment file with secondary structure data.
    #
    # Sequence alignment files are of two types:
    # Type 1: Single protein (no alignment information, just STRIDE SS)
    # Type 2: Two proteins (BLAST alignment and STRIDE SS)
    #
    # Type 1
    # ------
    #
    # Lines beginning with '#' are comments
    # Line 1:   FASTA sequence of protein with one-letter codes
    # Line 2:   STRIDE secondary structure information with one-letter codes
    #           H: alpha helix
    #           G: 3_10 helix
    #           I: pi-helix
    #           E: beta-strand
    #           B or b: bridge
    #           T: turn
    #           . or C: coil
    #
    # Type 2
    # ------
    #
    # Lines beginning with '#' are comments
    # Lines 1-3 are for protein 1, lines 4-6 are for protein 2
    # Line 1 or 4:  FASTA sequence of protein with one-letter codes. '-'
    #               represents an alignment gap
    # Line 2 or 5:  STRIDE secondary structure information with one-letter codes
    #               H: alpha helix
    #               G: 3_10 helix
    #               I: pi-helix
    #               E: beta-strand
    #               B or b: bridge
    #               T: turn
    #               #. or C: coil
    #               -: alignment gap
    # Line 3 or 6:  Mutation information
    #               . => no mutation
    #               - => alignment gap
    ############################################################################
    stride_parser = {'H': r"$\alpha$-Helix",
                     'G': r"$3_{10}$-Helix",
                     'I': r"$\pi$-Helix",
                     'E': r"$\beta$-Strand",
                     'B': r"Bridge",
                     'b': r"Bridge",
                     'T': r"Turn",
                     '.': r"Coil",
                     "C": r"Coil"}
    stride_colors = {'H': 'b',
                     'G': 'g',
                     'I': 'r',
                     'E': 'c',
                     'B': 'm',
                     'b': 'm',
                     'T': 'y',
                     '.': 'k',
                     'C': 'k'}

    with open(seq_file) as f:
        lines = []
        for line in f:
            if line.strip()[0] != '#':
                lines.append(line.strip())

    # Check that all line lengths match
    seq_len = len(lines[0])
    for line in lines:
        assert(len(line) == seq_len)

    ############################################################################
    # Single protein
    ############################################################################
    if nprot == 1:
        # Parse sequence data
        restype_dict = {}
        ss_dict = {}

        seq = lines[0]
        ss = lines[1]

        for i in range(seq_len):
            restype_dict[i] = seq[i]
            ss_dict[i] = ss[i]

        # Load structure into MDA
        u = mda.Universe(structfs[0])
        mda_select = OPA().selection_parser["alpha_C"]
        u_select = u.select_atoms(mda_select)
        resids = [atom.residue.resid for atom in u_select]

        # Begin analysis
        name = names[0]
        calc_dir = calc_dirs[0]
        di_format = di_formats[0]

        tsa = TimeSeriesAnalysis()

        RMSF = []

        # Unfold-fold
        for phi_idx, phi in enumerate(phivals):
            RMSF_phi = []
            for run_idx, run in enumerate(runs):
                dev_unfold = tsa.load_TimeSeries(calc_dir + di_format.format(phi=phi, run=run))
                RMSF_run = np.sqrt(np.mean(dev_unfold[start_time:].data_array ** 2, axis=0))
                RMSF_phi.append(RMSF_run)
            RMSF.append(RMSF_phi)

        RMSF = np.array(RMSF)

        # (phi, run, heavy atom)
        print(RMSF.shape)

        RMSF_mean = RMSF.mean(axis=1)
        RMSF_std = RMSF.std(axis=1)

        RMSF_native_mean = RMSF_mean[0, :].mean()
        RMSF_native_std = RMSF_mean[0, :].std()

        for phi_idx, phi in enumerate(tqdm(phivals)):
            fig, ax = plt.subplots(figsize=(16, 8), dpi=500)

            xvals = np.array(range(seq_len))

            yvals = RMSF_mean[phi_idx]
            yerrs = RMSF_std[phi_idx]

            ax.errorbar(xvals, yvals, yerr=yerrs, fmt='s', label=name, capsize=2.0)

            if plot_native:
                ax.axhline(y=RMSF_native_mean, label=r"$\phi = {}$".format(phivals[0]), color="green")
                ax.fill_between(xvals, RMSF_native_mean - RMSF_native_std, RMSF_native_mean + RMSF_native_std, color="green", alpha=0.4)

            xticks = xvals
            xticklabels = ['{}{}'.format(restype_dict[residx], resids[residx]) for residx in range(seq_len)]
            xtickcolors = [stride_colors[ss_dict[residx]] for residx in range(seq_len)]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Residue")

            for idx, t in enumerate(ax.xaxis.get_ticklabels()):
                t.set_color(xtickcolors[idx])
                t.set_fontsize(8)

            ax.set_ylabel(r"RMSF ($\AA$)")
            ax.set_title(r"$\phi = {}$ kJ/mol".format(phi))

            patchlist = []
            for ss_type in stride_colors.keys():
                if ss_type != '.' and ss_type != 'b':
                    patchlist.append(mpatches.Patch(color=stride_colors[ss_type], label=stride_parser[ss_type]))

            ss_legend = plt.legend(handles=patchlist, loc='upper left')
            ax.add_artist(ss_legend)

            plt.legend(loc='upper right')

            ax.grid()

            if set_max:
                ax.set_ylim([0, np.max(RMSF_mean) + np.max(RMSF_std)])

            plt.savefig(imgformat.format(phi=phi), bbox_inches='tight')

            plt.savefig(imgformat.format(phi="movie.{:05d}".format(phi_idx)), bbox_inches='tight')

            plt.close()

    ############################################################################
    # Compare proteins
    ############################################################################
    elif nprot == 2:
        restype_dicts = []  # list of residue type dictionaries, one for each protein
        alignment_dicts = []  # list of alignment dictionaries, one for each protein
        ss_dicts = []  # list of secondary structure dictionaries, one for each protein
        mutation_dicts = []  # list of mutation dictionaries, one for each protein

        resid_prots = []  # residue IDs for each protein

        line_len = len(lines[0])

        for prot_idx in range(nprot):
            # Parse sequence data
            restype_dict = {}
            alignment_dict = {}
            ss_dict = {}
            mutation_dict = {}

            aligned_seq = lines[3 * prot_idx]
            aligned_ss = lines[3 * prot_idx + 1]
            aligned_mut = lines[3 * prot_idx + 2]

            resctr = 0
            for i in range(line_len):
                if aligned_seq[i] != '-':
                    restype_dict[resctr] = aligned_seq[i]
                    alignment_dict[resctr] = i
                    ss_dict[resctr] = aligned_ss[i]
                    if aligned_mut[i] == '.':
                        mutation_dict[resctr] = False
                    else:
                        mutation_dict[resctr] = True
                    resctr += 1

            restype_dicts.append(restype_dict)
            alignment_dicts.append(alignment_dict)
            ss_dicts.append(ss_dict)
            mutation_dicts.append(mutation_dict)

            # Load structure into MDA
            u = mda.Universe(structfs[prot_idx])
            mda_select = OPA().selection_parser["alpha_C"]
            u_select = u.select_atoms(mda_select)
            u_resid = [atom.residue.resid for atom in u_select]
            resid_prots.append(u_resid)

        align_len = line_len

        RMSF_mean_prots = []
        RMSF_std_prots = []
        RMSF_native_mean_prots = []
        RMSF_native_std_prots = []

        for prot_idx in range(nprot):
            # Begin analysis
            name = names[prot_idx]
            calc_dir = calc_dirs[prot_idx]
            di_format = di_formats[prot_idx]

            tsa = TimeSeriesAnalysis()

            RMSF = []

            # Unfold-fold
            for phi_idx, phi in enumerate(phivals):
                RMSF_phi = []
                for run_idx, run in enumerate(runs):
                    dev_unfold = tsa.load_TimeSeries(calc_dir + di_format.format(phi=phi, run=run))
                    RMSF_run = np.sqrt(np.mean(dev_unfold[start_time:].data_array ** 2, axis=0))
                    RMSF_phi.append(RMSF_run)
                RMSF.append(RMSF_phi)

            RMSF = np.array(RMSF)

            # (phi, run, heavy atom)
            print(RMSF.shape)

            RMSF_mean = RMSF.mean(axis=1)
            RMSF_std = RMSF.std(axis=1)
            RMSF_mean_prots.append(RMSF_mean)
            RMSF_std_prots.append(RMSF_std)

            RMSF_native_mean = RMSF_mean[0, :].mean()
            RMSF_native_std = RMSF_mean[0, :].std()
            RMSF_native_mean_prots.append(RMSF_native_mean)
            RMSF_native_std_prots.append(RMSF_native_std)

        RMSF_mean_max = max(np.max(RMSF_mean_prots[0]), np.max(RMSF_mean_prots[1]))
        RMSF_std_max = max(np.max(RMSF_std_prots[0]), np.max(RMSF_std_prots[1]))

        for phi_idx, phi in enumerate(tqdm(phivals)):
            fig, ax = plt.subplots(figsize=(16, 8), dpi=500)

            for prot_idx in range(nprot):
                RMSF_mean = RMSF_mean_prots[prot_idx]
                RMSF_std = RMSF_std_prots[prot_idx]

                prot_plot_len = len(RMSF_mean_prots[prot_idx][phi_idx])
                resids = [resid_prots[prot_idx][i] for i in range(prot_plot_len)]

                xvals = [alignment_dicts[prot_idx][i] for i in range(len(resids))]
                yvals = RMSF_mean_prots[prot_idx][phi_idx]
                yerrs = RMSF_std_prots[prot_idx][phi_idx]

                ax.errorbar(xvals, yvals, yerr=yerrs, fmt='s', label=names[prot_idx], capsize=2.0)

                # IMPORTANT: Modify when changing from alpha_C to other type
                xticks = xvals
                xticklabels = ['{}{}'.format(restype_dicts[prot_idx][residx], resids[residx]) for residx in range(len(xticks))]
                xtickcolors = [stride_colors[ss_dicts[prot_idx][residx]] for residx in range(len(xticks))]
                xtickweights = []
                for residx in range(len(xticks)):
                    if(mutation_dicts[prot_idx][residx]):
                        xtickweights.append("bold")
                        xticklabels[residx] += "*"
                    else:
                        xtickweights.append("normal")

                if(prot_idx == 0):
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation=90)
                    ax.set_xlabel(names[prot_idx])

                    for idx, t in enumerate(ax.xaxis.get_ticklabels()):
                        t.set_color(xtickcolors[idx])
                        t.set_weight(xtickweights[idx])
                        t.set_fontsize(8)

                elif(prot_idx == 1):
                    secax = ax.secondary_xaxis('top')
                    secax.set_xticks(xticks)
                    secax.set_xticklabels(xticklabels, rotation=90)
                    secax.set_xlabel(names[prot_idx])

                    for idx, t in enumerate(secax.xaxis.get_ticklabels()):
                        t.set_color(xtickcolors[idx])
                        t.set_weight(xtickweights[idx])
                        t.set_fontsize(8)
                else:
                    raise NotImplementedError()

            ax.set_ylabel("RMSF")
            ax.set_title(r"$\phi = {}$ kJ/mol: P-unfold".format(phi))

            patchlist = []
            for ss_type in stride_colors.keys():
                patchlist.append(mpatches.Patch(color=stride_colors[ss_type], label=stride_parser[ss_type]))

            ss_legend = plt.legend(handles=patchlist, loc='upper left')
            ax.add_artist(ss_legend)

            plt.legend(loc='upper right')

            ax.set_xlim([0, align_len])

            ax.grid()

            if set_max:
                ax.set_ylim([0, RMSF_mean_max + RMSF_std_max])

            plt.savefig(imgformat.format(phi=phi), bbox_inches='tight')

            plt.savefig(imgformat.format(phi="movie.{:05d}".format(phi_idx)), bbox_inches='tight')

            plt.close()

    else:
        raise ValueError("Invalid number of proteins. This script can work with either one or two proteins.")

    if make_movie:
        os.system("ffmpeg -r 0.5 -i {} -vcodec mpeg4 -y -vb 200M -q:v 1 {}".format(imgformat.format(phi="movie.%05d"), moviefile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSF average across runs for each phi value for one or two proteins.")
    parser.add_argument("-nprot", type=int, help="number of proteins (1 or 2)")
    parser.add_argument("-structfs", type=str, nargs='+', help="path to structure files (.pdb, .gro, or .tpr) for each protein (space separated)")
    parser.add_argument("-names", type=str, nargs='+', help="names of proteins (space separated)")
    parser.add_argument("-seqfile", type=str, help="sequence alignment file")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dirs", type=str, nargs='+', help="directory containing hydration OPs extracted by INDUSAnalysis, one for each protein (space separated)")
    parser.add_argument("-di_formats", type=str, nargs='+', help="format of .pkl file containing residue deviations, with {phi} placeholders for phi value and {run} placeholders for run value, one for each protein (space separated)")
    parser.add_argument("-imgformat", help="output image format, with {phi} placeholders for phi value")
    parser.add_argument("--plot_native", action='store_true', help="plot band indicating average RMSF in native state")
    parser.add_argument("--set_max", action='store_true', help="set a fixed maximum value on the y-axis, determined by the max RMSF across the simulation")
    parser.add_argument("--make_movie", action='store_true', help="stitch together image files with FFMPEG to make a movie")
    parser.add_argument("-moviefile", type=str, help="movie file name")

    a = parser.parse_args()

    RMSF(a.nprot, a.structfs, a.names, a.seqfile, a.phi, a.runs, a.start, a.calc_dirs, a.di_formats, a.imgformat, a.plot_native, a.set_max, a.make_movie, a.moviefile)
