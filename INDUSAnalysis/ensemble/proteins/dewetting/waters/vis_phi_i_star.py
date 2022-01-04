"""
Generates PDB files with indicators for whether an atom has passed its phi_i*
value to visualize protein dewetting.
"""
import argparse
import pickle

import numpy as np

import MDAnalysis as mda

from tqdm import tqdm


def vis_phi_i_star(pklfile, structfile, trajfile, phi, pdb, buried_surface_indicator):
    ############################################################################
    # Load data
    ############################################################################
    with open(pklfile, "rb") as infile:
        phi_i_star_data = pickle.load(infile)

    phi_i_stars = phi_i_star_data['phi_i_stars']

    with open(buried_surface_indicator, "rb") as infile:
        buried_surface_indicator = np.load(infile)

    ############################################################################
    # Static PDB generation
    ############################################################################

    dewetted_at_or_before = []

    protein_sel = "protein"
    protein_heavy_sel = "protein and not name H*"

    # Load equilibrium simulation universe (to manipulate and write)
    u = mda.Universe(structfile, trajfile)
    u.add_TopologyAttr('tempfactors')

    phi_static_vals = np.linspace(phi[0], phi[1], int(phi[2]))
    pbar = tqdm(total=len(phi_static_vals))

    with mda.Writer(pdb, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        # indicator is buried = 1, surface = 0
        # invert to surface = 1, buried = 0
        mask = 1 - buried_surface_indicator

        u_protein = u.select_atoms(protein_sel)
        u_protein_heavy = u.select_atoms(protein_heavy_sel)
        u_protein_heavy.atoms.tempfactors = mask

        PDB.write(u_protein.atoms)
        pbar.update(1)

        for idx, phi in enumerate(phi_static_vals):
            # Select protein from current u
            u_protein = u.select_atoms(protein_sel)

            # If phi_i_star is greater than current phi => needs more bias to undergo transition
            # Then assign value = 1 + 1 = 2 (not yet dewetted)
            # Else (i.e. phi_i_star >= phi => transition occurs at or before current phi
            # Then assign value = 1 (dewetted)
            phi_i_star_premask = 1 + np.array(phi_i_stars > float(phi)).astype(np.float)

            # Assign non-zero value only if atom is a surface atom
            phi_i_star_mask = mask * phi_i_star_premask

            # Assign values to protein heavy atoms
            u_protein_heavy = u.select_atoms(protein_heavy_sel)
            u_protein_heavy.atoms.tempfactors = phi_i_star_mask

            # Record atom ids
            # buried = 0
            # dewetted = 1
            # not yet dewetted = 2
            dewetted_at_or_before.append(set(np.argwhere(phi_i_star_mask == 1).flatten().tolist()))

            # Write to PDB as a new frame
            PDB.write(u_protein.atoms)
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-pklfile", help="output file to dump phi_i* data to (.pkl)")
    parser.add_argument("-structfile", help="path to structure file (.pdb, .gro, .tpr)")
    parser.add_argument("-trajfile", help="trajectory file (.pdb, .gro, .tpr) to extract first frame from for visualization")
    parser.add_argument("-phi", nargs=3, type=float, help="parameters to choose phi values for static PDB (start end spacing)")
    parser.add_argument("-pdb", help="static output PDB file")
    parser.add_argument("-buried_surface_indicator", help="file containing binary array classifying each heavy atom as a buried or surface atom (.pkl)")
    a = parser.parse_args()

    vis_phi_i_star(a.pklfile, a.structfile, a.trajfile, a.phi, a.pdb, a.buried_surface_indicator)
