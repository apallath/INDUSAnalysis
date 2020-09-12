#!/bin/bash

# Script to run a waters post-processing calculation using INDUSAnalysis

# Arguments for run_indus_waters [Comment out]
args=(
# GROMACS-INDUS output file
/path/to/phiout.dat
# GROMACS structure (.gro)/portable run (.tpr) file
/path/to/struct.gro
# GROMACS trajectory file
/path/to/traj.xtc

# Radius of spherical probe volume (in A)
-radius 6.0
# Interval (number of frames) to read trajctory when performing calculations
-skip 2

# Window (number of frames) for calculating smoothed averages
-window 50

# Generate PDB with heavy atom waters stored in bfactors
#--genpdb

# Prefix of output image and data files
-opref indus
# Output format of image files
-oformat png
# DPI of image files
-dpi 300

# Replot from existing data (do not perform calculations)
#--replot
# Prefix of calculation files to replot data from
#-replotpref indus

# Show Matplotlib plots
#--show
# Perform calculations on remote cluster [=> use text-only Matplotlib backend]
--remote
# Output
--verbose
)

# INDUSAnalysis root directory
root_dir=~/analysis_scripts

set -x

python -u $root_dir/scripts/run_indus_waters.py "${args[@]}"
