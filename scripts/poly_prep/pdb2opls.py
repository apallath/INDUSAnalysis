"""
Reads PDB file and an input file matching PDB atom types to GROMACS OPLS codes to generate the following:
- GROMACS structure file (.gro) 
- GROMACS Molecule topology file (.itp)
"""

import argparse

import MDAnalysis as mda

parser = argparse.ArgumentParser()
parser.add_argument("pdbname", help="PDB file name/path")
parser.add_argument("gmx_opls_path", help="path to GROMACS oplsaa.ff")
parser.add_argument("atom_opls_index", help="index matching PDB atom types to GROMACS opls code")
