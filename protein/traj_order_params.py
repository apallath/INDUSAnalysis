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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("structf", help="Structure file (.gro)")
    parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")

    sf = open(args.structf)
    tf = open(args.trajf)
