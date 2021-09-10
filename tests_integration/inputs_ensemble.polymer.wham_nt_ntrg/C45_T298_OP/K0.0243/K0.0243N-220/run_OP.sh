#!/bin/bash

#Job identifier
#SBATCH -J OP_-220

#partition to use
#SBATCH -p standby

#Request resources
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

#Wall clock limit: HH:MM:SS
#SBATCH -t 00:15:00

#output and error files for run
#%x - jobname
#%N - short hostname
#%j - jobid of running job
#
#SBATCH -o %x.out.%N.%j
#SBATCH -e %x.err.%N.%j

# Arguments for run_protein_order_params [Comment out]
args=(
# GROMACS structure (.gro)/portable run (.tpr) file
../../../Inputs/C45/conf.gro
# GROMACS trajectory file
../../../C45_T298/K0.0243/K0.0243N-220/ofile_mol.xtc
# Reference trajectory file for alignment [default: same as trajectory file]
# -reftrajf
# Timestep to read reference structure from
-reftstep 1

# Selection group to use for aligning structures
-align alkane_ua
# Selection group to calculate Rg and RMSD for
-select alkane_ua
# Interval (number of frames) to read trajctory when performing calculations
#-skip 2

# Window (number of frames) for calculating smoothed averages
-window 50

# Generate PDB with per-atom deviations
#--genpdb

# Prefix of output image and data files
-opref C45
# Output format of image files
-oformat png
# DPI of image files
-dpi 150

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

set -xe

python -u $root_dir/scripts/run_protein_order_params.py "${args[@]}" >> OP.out
