#!/bin/bash

# Script to run a protein structural order parameters post-processing calculation
# using INDUSAnalysis

# Arguments for run_protein_order_params [Comment out]
args=(
# GROMACS structure (.gro)/portable run (.tpr) file
/path/to/prod.gro
# GROMACS trajectory file
/path/to/traj.xtc
# Reference trajectory file for alignment [default: same as trajectory file]
#-reftrajf /path/to/reftraj.xtc
# Timestep to read reference structure from
-reftstep 0

# Selection group to use for aligning structures
-align backbone
# Selection group to calculate RMSD for
-select heavy
# Interval (number of frames) to read trajctory when performing calculations
-skip 2

# Window (number of frames) for calculating smoothed averages
-window 50

# Generate PDB with per-atom deviations
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

python -u $root_dir/scripts/run_protein_order_params.py "${args[@]}"
