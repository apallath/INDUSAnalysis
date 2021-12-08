"""
Plots phi-series and generates pie-chart movies of
A) Hydration with phi, of
- atoms from charged or hydrophobic or hydrophilic residues
- polar or nonpolar atoms (as per the Kapcha-Rossky scale)
- buried or surface atoms
B) Hydration of atoms belonging to different secondary structure classes
C) Hydration of atoms belonging to different secondary structure groups
"""
import os
import pickle
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm

import MDAnalysis as mda

# Use text-only Matplotlib backend
matplotlib.use('Agg')


###############################################################################
# Functions returning values to plot on pie chart movie frames
###############################################################################

def charged_polar_buried_valueslist(indices, protein_heavy, buried, kr_scale):
    hydrophobic = ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'PRO', 'TRP']

    n_res_charged = 0
    n_res_hydrophobic = 0
    n_res_hydrophilic = 0

    for res in protein_heavy.atoms[indices].residues:
        if np.abs(res.charge) > 0.5:
            n_res_charged += 1
        elif res.resname in hydrophobic:
            n_res_hydrophobic += 1
        else:
            n_res_hydrophilic += 1

    n_atoms_polar_buried = 0
    n_atoms_polar_surface = 0
    n_atoms_non_polar_buried = 0
    n_atoms_non_polar_surface = 0

    for heavyidx in indices:
        atom = protein_heavy.atoms[heavyidx]
        atom_res = atom.residue.resname
        atom_name = atom.name
        if atom_res in ["MET"] and atom_name in ["OC1", "OC2"]:
            atom_name = "O"

        atom_polar = kr_scale.get("{} {}".format(atom_res, atom_name))
        if atom_polar == 1 and buried[heavyidx]:
            n_atoms_polar_buried += 1
        elif atom_polar == 1 and not buried[heavyidx]:
            n_atoms_polar_surface += 1
        elif atom_polar == -1 and buried[heavyidx]:
            n_atoms_non_polar_buried += 1
        elif atom_polar == -1 and not buried[heavyidx]:
            n_atoms_non_polar_surface += 1
        else:
            print(buried[heavyidx])
            print("Exception at {} {}".format(atom_res, atom_name))

    return(np.array([n_res_charged, n_res_hydrophobic, n_res_hydrophilic]),
           np.array([n_atoms_polar_surface, n_atoms_polar_buried, n_atoms_non_polar_surface, n_atoms_non_polar_buried]))


def sec_struct_class_valueslist(indices, protein_heavy, stride_dict):
    mapper = {"H": 0, "G": 1, "I": 2, "E": 3, "T": 4, "B": 5, "C": 6}
    comp = [0, 0, 0, 0, 0, 0, 0]

    for heavyidx in indices:
        atom = protein_heavy.atoms[heavyidx]
        resid = atom.residue.resid
        comp[mapper[stride_dict[resid + 1]]] += 1

    return(np.array(comp))


def sec_struct_group_valueslist(indices, protein_heavy, all_groups, stride_group):
    idx_to_group = dict()
    group_to_idx = dict()

    for gidx, gname in enumerate(all_groups):
        idx_to_group[gidx] = gname
        group_to_idx[gname] = gidx

    comp = len(all_groups) * [0]

    for i in indices:
        atomi = protein_heavy.atoms[i]
        residi = atomi.residue.resid
        atomi_group = stride_group[residi + 1]

        comp[group_to_idx[atomi_group]] += 1

    return(np.array(comp))


###############################################################################
# Main calculation function
###############################################################################

def hydrated_atom_chars():
    pass


if __name__ == "__main__":
    hydrated_atom_chars()
