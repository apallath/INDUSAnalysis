"""
Performs buried v/s surface classification for the entire phi-ensemble.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm

from INDUSAnalysis.timeseries import TimeSeriesAnalysis


def buried_surface_ensemble_classifier(phivals: list,
                                       runs: list,
                                       start_time: int,
                                       structfile: str,
                                       ni_format: str,
                                       buried_cut: float,
                                       trajformat: str,
                                       classfile: str,
                                       classpdb: str):
    tsa = TimeSeriesAnalysis()

    ############################################################################
    # Load data
    ############################################################################
    u = mda.Universe(structfile)
    protein_heavy = u.select_atoms("protein and not name H*")
    protein_heavy_indices = protein_heavy.atoms.indices

    meanwaters = np.zeros((len(phivals), len(runs), len(protein_heavy_indices)))

    # All phi values
    for idx, phi in enumerate(phivals):
        for runidx, run in enumerate(runs):
            ts = tsa.load_TimeSeries(ni_format.format(phi=phi, run=run))
            ts = ts[start_time:]
            run_waters = ts.data_array[:, protein_heavy_indices]

            # Calculate per-atom mean waters and var waters for each run
            mean_run_waters = np.mean(run_waters, axis=0)

            # Append per-atom mean and var waters for each run
            meanwaters[idx, runidx, :] = mean_run_waters

    mean_meanwaters = np.mean(meanwaters, axis=1)

    # Calculate buried/surface classes at each phi
    # 1 = surface
    # 0 = buried
    classes = mean_meanwaters > buried_cut
    np.save(classfile, classes)

    ############################################################################
    # Dynamic PDB generation
    ############################################################################

    protein_sel = "protein"
    backbone_sel = "name CA or name C or name N"
    protein_heavy_sel = "protein and not name H*"

    # Load reference equilibrium simulation universe
    uref = mda.Universe(structfile, trajformat.format(phi=0))
    uref.trajectory[0]

    uref_protein = uref.select_atoms(protein_sel)
    uref_backbone = uref.select_atoms(backbone_sel)

    # Load equilibrium simulation universe (to manipulate and write)
    u = mda.Universe(structfile, trajformat.format(phi=0))
    u.add_TopologyAttr('tempfactors')

    pbar = tqdm(total=len(phivals))

    with mda.Writer(classpdb, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:

        # Originally buried = 0
        # Originally surface = 1
        original_mask = classes[0]

        u_protein = u.select_atoms(protein_sel)
        u_protein_heavy = u.select_atoms(protein_heavy_sel)

        PDB.write(u_protein.atoms)
        pbar.update(1)

        for idx, phi in enumerate(phivals):
            # Load universe
            upos = mda.Universe(structfile, trajformat.format(phi=phi))

            # Skip to middle frame
            upos.trajectory[int(len(upos.trajectory) / 2)]

            # Zero COM
            upos_protein = upos.select_atoms(protein_sel)
            upos_protein.atoms.positions = upos_protein.atoms.positions - upos_protein.center_of_mass()

            # Rotate to align backbone with uref backbone
            upos_backbone = upos.select_atoms(backbone_sel)
            R, rmsd = align.rotation_matrix(upos_backbone.atoms.positions,
                                            uref_backbone.atoms.positions - uref_protein.center_of_mass())
            upos_protein.atoms.rotate(R)

            # Translate to align with COM of uref
            upos_protein.atoms.positions = upos_protein.atoms.positions + uref_protein.center_of_mass()

            # Select protein from current u
            u_protein = u.select_atoms(protein_sel)

            # Assign RMSD-minimized positions
            u_protein.atoms.positions = upos_protein.atoms.positions

            # Mask calculations
            # Buried and now buried = 0 + 0 = 0
            # Surface and now buried = 1 + 0 = 1
            # Buried and now surface = 0 + 2 = 2
            # Surface and now surface = 1 + 2 = 3
            mask = original_mask + 2 * classes[idx]

            # Assign values to protein heavy atoms
            u_protein_heavy = u.select_atoms(protein_heavy_sel)
            u_protein_heavy.atoms.tempfactors = mask

            # Write to PDB as a new frame
            PDB.write(u_protein.atoms)
            pbar.update(1)


################################################################################
# Buried and now buried = 0 + 0 = 0
# Surface and now buried = 1 + 0 = 1
# Buried and now surface = 0 + 2 = 2
# Surface and now surface = 1 + 2 = 3
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-structfile", help="path to structure file (.pdb, .gro, .tpr)")
    parser.add_argument("-ni_format", help="format of .pkl file containing Ntw, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-buried_cut", type=int, help="cutoff folded state <ni> at or below which (<=) atom i is classified as a buried atom")
    parser.add_argument("-trajformat", help="format for trajectory file (.pdb, .gro, .tpr) with {phi} placeholder for phi value")
    parser.add_argument("-classfile", default="buried_surface_indicator.npy", help="output numpy file buried v/s surface classes for heavy atoms(default=buried_surface_indicator.npy)")
    parser.add_argument("-classpdb", default="buried_surface_indicator.pdb", help="output PDB buried v/s surface classes as bfactors (default=buried_surface_indicator.pdb)")

    a = parser.parse_args()

    buried_surface_ensemble_classifier(a.phi, a.runs, a.start, a.structfile, a.ni_format, a.buried_cut, a.trajformat, a.classfile, a.classpdb)
