#!/bin/bash

# Script to run contacts post-processing calculation
# using INDUSAnalysis

# Arguments for run_protein_order_params [Comment out]
args=(
# GROMACS structure (.gro)/portable run (.tpr) file
/path/to/prod.tpr
# GROMACS trajectory file
/path/to/traj.xtc

# Method for calculating contacts
-method atomic-sh
# Distance cutoff for contacts calculation (in A)
-distcutoff 7.0
# Connectivity threshold for contacts (definition varies by method)
# For example, for atomic_sh, connectivity threshold is the minmum number of
# bonds along the shortest path on the bond network which two atoms have to be
# separated by to form a contact
-connthreshold 5
# Interval (number of frames) to read trajctory when performing calculations
-skip 2
# Number of bins for histogram
-bins 20

# Time to begin averaging at (in ps)
-obsstart 500
# Time to stop averaging at (in ps)
# -obsend 4000

# Prefix of output image and data files
-opref indus
# Output format of image files
-oformat png
# DPI of image files
-dpi 300

# Show Matplotlib plots
# --show
# Perform calculations on remote cluster [=> use text-only Matplotlib backend]
--remote
# Output
--verbose
)

# INDUSAnalysis root directory
root_dir=~/analysis_scripts

set -x

python -u $root_dir/scripts/run_contacts.py "${args[@]}"
