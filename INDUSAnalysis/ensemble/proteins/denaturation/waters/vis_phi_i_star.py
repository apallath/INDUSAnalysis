"""
Plots ni v/s phi and phi_i* for a set of representative atoms, and also for each atom i.

Stores calculated phi_i* values.
"""
import argparse
import os
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator

import MDAnalysis as mda
from MDAnalysis.analysis import align

from tqdm import tqdm


def vis_phi_i_star(pklfile, structfile, trajformat, phi_dyn, phi_static, pdb_dyn,
                   pdb_static, infty_cutoff):
    ############################################################################
    # Load data
    ############################################################################
    with open(pklfile, "rb") as infile:
        phi_i_star_data = pickle.load(infile)

    phi_i_stars = phi_i_star_data['phi_i_stars']
    # phi_i_star_errors = phi_i_star_data['phi_i_star_errors']
    # delta_ni_trans = phi_i_star_data['delta_ni_trans']
    # delta_phi_trans = phi_i_star_data['delta_phi_trans']

    ############################################################################
    # Dynamic PDB generation
    ############################################################################
    wetted_at_or_before = []

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

    pbar = tqdm(total=len(phi_dyn))

    with mda.Writer(pdb_dyn, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        mask = 2 * (phi_i_stars < infty_cutoff)

        u_protein = u.select_atoms(protein_sel)
        u_protein_heavy = u.select_atoms(protein_heavy_sel)
        u_protein_heavy.atoms.tempfactors = mask

        PDB.write(u_protein.atoms)
        pbar.update(1)

        for idx, phi in enumerate(phi_dyn):
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

            # If phi_i_star is more negative than current phi => needs more bias to undergo transition
            # Then assign value = 1 + 1 = 2
            # Else (i.e. phi_i_star >= phi => transition occurs at or before current phi
            # Then assign value = 1
            phi_i_star_premask = 1 + np.array(phi_i_stars < phi).astype(np.float)

            # If phi_i_star >= infty_cutoff => discarded from start
            # Then assign value = 0
            # => Assign non-zero value only if phi_i_star < infty_cutoff
            mask = phi_i_stars < infty_cutoff
            phi_i_star_mask = mask * phi_i_star_premask

            # Assign values to protein heavy atoms
            u_protein_heavy = u.select_atoms(protein_heavy_sel)
            u_protein_heavy.atoms.tempfactors = phi_i_star_mask

            # Record atom ids
            wetted_at_or_before.append(set(np.argwhere(phi_i_star_mask == 1).flatten().tolist()))

            # Write to PDB as a new frame
            PDB.write(u_protein.atoms)
            pbar.update(1)

    ############################################################################
    # Static PDB generation
    ############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-pklfile", help="output file to dump phi_i* data to (.pkl)")
    parser.add_argument("-structfile", help="path to structure file (.pdb, .gro, .tpr)")
    parser.add_argument("-trajformat", help="format for trajectory file (.pdb, .gro, .tpr) with {phi} placeholder for phi value")
    parser.add_argument("-phi_dyn", type=str, nargs='+', help="phi values to write to dynamic PDB")
    parser.add_argument("-phi_static", nargs=3, type=float, help="parameters to choose phi values for static PDB (start end spacing)")
    parser.add_argument("-pdb_dyn", help="dynamic output PDB file")
    parser.add_argument("-pdb_static", help="static output PDB file")
    parser.add_argument("-infty_cutoff", type=float, help="upper limit on phi_i* beyond which an atom is considered always hydrated (phi_i* = infinity)")
    a = parser.parse_args()

    vis_phi_i_star(a.pklfile, a.structfile, a.trajformat, a.phi_dyn, a.phi_static, a.pdb_dyn, a.pdb_static, a.infty_cutoff)
