"""
Plots unfolding and folding RMSFs of one or two protein structures, given BLAST alignment with STRIDE secondary structures.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from INDUSAnalysis.protein_order_params import OrderParamsAnalysis as OPA
from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis
import MDAnalysis as mda
from tqdm import tqdm

def RMSF(nprot, seq_file):
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

    # Read in lines
    if nprot == 1:
        restype_dict = {}
        ss_dict = {}

        seq = lines[0]
        ss = lines[1]

        for i in range(seq_len):
            restype_dict[i] = seq[i]
            ss_dict[i] = ss[i]

    elif nprot == 2:
        for prot_idx in range(NPROT):
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

        align_len = line_len

    else:
        raise ValueError("Invalid number of proteins. This script can work with either one or two proteins.")

    """
    tsa = TimeSeriesAnalysis()

    align = "backbone"
    selection = "alpha_C"
    phivals = [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, -0.5, -1, -1.25, -1.5, -1.75, -2, -2.05, -2.1, -2.15, -2.2, -2.25,
               -2.3, -2.4, -2.5, -2.75, -3, -3.5, -4, -4.25, -5, -6, -7, -7.5, -8, -9, -10]
    runs = [1, 2, 3, 4, 5]
    start_time = 500
    end_time = None

    TPR_DIR = "/home/akash/Documents/pressure_denaturation/ubiquitin/"
    CALC_DIR = "/home/akash/Documents/pressure_denaturation/ubiquitin_analysis/"
    NAME = r"Ubiquitin"

    # Load structure into MDA
    u = mda.Universe(TPR_DIR + "alt_tpr/indus.tpr")
    mda_select = OPA().selection_parser[selection]
    u_select = u.select_atoms(mda_select)

    assert(len(u_select.atoms) == seq_len)

    # unfold          each phi      each select atom     each run
    RMSFu = np.zeros((len(phivals), len(u_select.atoms), len(runs)))
    # unfold          each phi      each select atom     each run
    RMSFf = np.zeros((len(phivals), len(u_select.atoms), len(runs)))

    # Unfold-fold
    for phi_idx, phi in enumerate(phivals):
        for run in runs:
            dev_unfold = tsa.load_TimeSeries(CALC_DIR + "unfold_fold/OP/i{}/{}/unfold_deviations_{}_{}.pkl".format(phi, run, align, selection))
            RMSFu_run = np.sqrt(np.mean(dev_unfold[start_time:end_time].data_array ** 2, axis=0))
            RMSFu[phi_idx, :, run - 1] = RMSFu_run

            dev_fold = tsa.load_TimeSeries(CALC_DIR + "unfold_fold/OP/i{}/{}/fold_deviations_{}_{}.pkl".format(phi, run, align, selection))
            RMSFf_run = np.sqrt(np.mean(dev_fold[start_time:end_time].data_array ** 2, axis=0))
            RMSFf[phi_idx, :, run - 1] = RMSFf_run

    RMSFu_mean = RMSFu.mean(axis=2)
    RMSFu_std = RMSFu.std(axis=2)

    RMSFf_mean = RMSFf.mean(axis=2)
    RMSFf_std = RMSFf.std(axis=2)

    for phi_idx, phi in enumerate(tqdm(phivals)):
        fig, ax = plt.subplots(figsize=(16, 8), dpi=500)

        xvals = np.array(range(seq_len))

        yvals = RMSFu_mean[phi_idx]
        yerrs = RMSFu_std[phi_idx]

        ax.errorbar(xvals, yvals, yerr=yerrs, fmt='s', label=NAME, capsize=2.0)

        # IMPORTANT: Modify when changing from alpha_C to other type
        xticks = xvals
        xticklabels = ['{}{}'.format(restype_dict[resid], resid) for resid in range(seq_len)]
        xtickcolors = [stride_colors[ss_dict[resid]] for resid in range(seq_len)]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_xlabel(NAME)

        for idx, t in enumerate(ax.xaxis.get_ticklabels()):
            t.set_color(xtickcolors[idx])
            t.set_fontsize(8)

        ax.set_ylabel("RMSF")
        ax.set_title(r"$\phi = {}$ kJ/mol: P-unfold".format(phi))

        patchlist = []
        for ss_type in stride_colors.keys():
            patchlist.append(mpatches.Patch(color=stride_colors[ss_type], label=stride_parser[ss_type]))

        ss_legend = plt.legend(handles=patchlist, loc='upper left')
        ax.add_artist(ss_legend)

        plt.legend(loc='upper right')

        ax.grid()

        plt.savefig("compare_unfold_{}.png".format(phi), bbox_inches='tight')

        ax.set_ylim([0, 30])

        plt.savefig("compare_unfold.{0:02}.png".format(phi_idx), bbox_inches='tight')

        plt.close()

        #######################
        # Fold
        #######################

        fig, ax = plt.subplots(figsize=(16, 8), dpi=500)

        xvals = np.array(range(seq_len))

        yvals = RMSFf_mean[phi_idx]
        yerrs = RMSFf_std[phi_idx]

        ax.errorbar(xvals, yvals, yerr=yerrs, fmt='s', label=NAME, capsize=2.0)

        # IMPORTANT: Modify when changing from alpha_C to other type
        xticks = xvals
        xticklabels = ['{}{}'.format(restype_dict[resid], resid) for resid in range(seq_len)]
        xtickcolors = [stride_colors[ss_dict[resid]] for resid in range(seq_len)]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_xlabel(NAME)

        for idx, t in enumerate(ax.xaxis.get_ticklabels()):
            t.set_color(xtickcolors[idx])
            t.set_fontsize(8)

        ax.set_ylabel("RMSF")
        ax.set_title(r"$\phi = {}$ kJ/mol: P-relax".format(phi))

        patchlist = []
        for ss_type in stride_colors.keys():
            patchlist.append(mpatches.Patch(color=stride_colors[ss_type], label=stride_parser[ss_type]))

        ss_legend = plt.legend(handles=patchlist, loc='upper left')
        ax.add_artist(ss_legend)

        plt.legend(loc='upper right')

        ax.grid()

        plt.savefig("compare_fold_{}.png".format(phi), bbox_inches='tight')

        ax.set_ylim([0, 30])

        plt.savefig("compare_fold.{0:02}.png".format(phi_idx), bbox_inches='tight')

    plt.close()
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSF average across runs for each phi value for one or two proteins.")
    parser.add_argument("-nprot", type=int, nargs='+', help="number of proteins (1 or 2)")
    parser.add_argument("-seq_align", type=str, help="sequence alignment file")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("-di_format", help="format of .pkl file containing backbone atom deviations, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-imgformat", help="output image format, with {phi} placeholders for phi value")

    a = parser.parse_args()

    RMSF(a.phi, a.runs, a.start, a.calc_dir, a.Ntw_format, a.imgfile, a.D_by_A_guess, a.E_guess, a.P0)
