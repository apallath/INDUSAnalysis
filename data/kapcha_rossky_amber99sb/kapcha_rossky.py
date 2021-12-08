"""Parses data extracted from GROMACS oplsaa.ff/aminoacids.rtp and calculates
Kapcha-Rossky hydrophobicities for each atom. Saves data in dictionary and
pickles to file

Link to paper: https://doi.org/10.1016/j.jmb.2013.09.039
"""

import numpy as np
import pickle

oplsaa_params_file = "oplsaa_res_atom_charges.txt"

kr_scale = {}

with open(oplsaa_params_file) as f:
    cur_res = ""
    for line in f:
        sl = line.strip()
        if sl == "":
            pass
        elif sl[0] == '[':
            cur_res = sl.split()[1]
        else:
            data = sl.split()
            atom_name = data[0]
            atom_charge = np.float(data[2])
            atom_polar = 0
            if np.abs(atom_charge) <= 0.25:
                atom_polar = -1
            else:
                atom_polar = 1
            kr_scale["{} {}".format(cur_res, atom_name)] = atom_polar

for key in kr_scale:
    if key.startswith("GLY"):
        print(key, kr_scale[key])

with open("kapcha_rossky_scale.pkl", "wb") as of:
    pickle.dump(kr_scale, of)
