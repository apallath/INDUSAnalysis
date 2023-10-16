"""
Generates PDB files with indicators for whether an atom has passed its phi_i*
value to visualize polymer dewetting.
"""
import argparse
import pickle

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm


def vis_phi_i_star(pklfile, structfile, trajfile, poly_selection, probe_selection, phi, pdb, spdb):
    ############################################################################
    # Load data
    ############################################################################
    with open(pklfile, "rb") as infile:
        phi_i_star_data = pickle.load(infile)

    phi_i_stars = phi_i_star_data['phi_i_stars']

    ############################################################################
    # Static PDB generation
    ############################################################################

    # Print color scale data range
    print(phi_i_stars.min(), phi_i_stars.max(), phi_i_stars.mean())

    # Load equilibrium simulation universe (to manipulate and write)
    u = mda.Universe(structfile, trajfile)
    u.add_TopologyAttr('tempfactors')

    with mda.Writer(spdb, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        u_poly = u.select_atoms(poly_selection)
        u_poly_probe = u.select_atoms(probe_selection)

        u_poly_probe.atoms.tempfactors = phi_i_stars

        PDB.write(u_poly.atoms)

    ############################################################################
    # Dynamic PDB generation
    ############################################################################

    dewetted_at_or_before = []

    # Load equilibrium simulation universe (to manipulate and write)
    u = mda.Universe(structfile, trajfile)
    u.add_TopologyAttr('tempfactors')

    phi_static_vals = np.linspace(phi[0], phi[1], int(phi[2]))
    pbar = tqdm(total=len(phi_static_vals))

    with mda.Writer(pdb, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        u_poly = u.select_atoms(poly_selection)
        u_poly_probe = u.select_atoms(probe_selection)

        for idx, phi in enumerate(phi_static_vals):
            # If phi_i_star is greater than current phi => needs more bias to undergo transition
            # Then assign value = 1 (not yet dewetted)
            # Else (i.e. phi_i_star >= phi => transition occurs at or before current phi
            # Then assign value = 0 (dewetted)
            phi_i_star_mask = np.array(phi_i_stars > float(phi)).astype(np.float)

            # Assign values to protein heavy atoms
            u_poly_probe.atoms.tempfactors = phi_i_star_mask

            # Record atom ids
            # dewetted = 0
            # not yet dewetted = 1
            dewetted_at_or_before.append(set(np.argwhere(phi_i_star_mask == 0).flatten().tolist()))

            # Write to PDB as a new frame
            PDB.write(u_poly.atoms)
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument(
        "-pklfile", default="phi_i_stars.pkl", help="output file to load phi_i* data from (.pkl, default=phi_i_stars.pkl)"
    )
    parser.add_argument("-structfile", help="path to structure file (.pdb, .gro, .tpr)")
    parser.add_argument("-trajfile", help="trajectory file (.pdb, .gro, .tpr) to extract first frame from for visualization")
    parser.add_argument("-poly_selection", help="selection string for polymer atoms")
    parser.add_argument("-probe_selection", help="selection string for probe atoms")
    parser.add_argument("-phi", nargs=3, type=float, help="parameters to choose phi values for static PDB (start end num)")
    parser.add_argument("-pdb", default="vis.pdb", help="dynamic output PDB file (default='vis.pdb')")
    parser.add_argument("-spdb", default="vis_static.pdb", help="static output PDB file (default='vis_static.pdb')")
    a = parser.parse_args()

    vis_phi_i_star(a.pklfile, a.structfile, a.trajfile, a.poly_selection, a.probe_selection, a.phi, a.pdb, a.spdb)
