"""
@author Akash Pallath

Analyse GROMACS trajectory files for a protein and calculate different
order parameters

Dependencies:
- argparse
- MDAnalysis
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.rms as mda_rms

def calc_Rg(u,selection):
    Rg = []
    sel = u.select_atoms(selection)
    for ts in u.trajectory:
        Rg.append((u.trajectory.time, sel.radius_of_gyration()))
    Rg = np.array(Rg)
    return Rg

def calc_RMSD(u,selection):
    RMSD = []
    sel = u.select_atoms(selection)
    #initial positions
    initpos = sel.positions.copy()
    for ts in u.trajectory:
        RMSD.append((u.trajectory.time, mda_rms.rmsd(initpos,sel.positions)))
    RMSD = np.array(RMSD)
    return RMSD

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("structf", help="Structure file (.gro)")
    parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
    parser.add_argument("-selection", help="atoms/groups to track order parameters for")
    args = parser.parse_args()
    selection = args.selection
    if selection == None:
        selection = 'protein'
    u = mda.Universe(args.structf,args.trajf)
    #radius of gyration
    sel_rg = calc_Rg(u,selection)
    #RMSD from initial structure
    sel_rmsd = calc_RMSD(u,selection)

    #store time series data
    np.save('Rg_{}.txt'.format(selection),sel_rg)
    np.save('RMSD_{}.txt'.format(selection),sel_rmsd)

    #plot
