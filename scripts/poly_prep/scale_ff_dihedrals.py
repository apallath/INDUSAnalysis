"""
Scales force field dihedrals.
"""
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-idir", help="Input FF directory")
parser.add_argument("-odir", help="Output FF directory")
parser.add_argument("-itpfile", help="Topolgy file to modify (default=ffbonded.itp)", default="ffbonded.itp")
parser.add_argument("-dihtype", help="""Dihedral potential to scale with '-' in place of whitespace 
(e.g. 'CH3-CH2-CH2-CH3' if you want to modify dihedral type 'CH3 CH2 CH2 CH3'). If 'all', then scales all dihedrals. (default=all)""", default="all")
parser.add_argument("-scale", type=float, help="Scaling factor s for each dihedral coefficient (i.e if original coeff is C, new coeff is s*C)")

args = parser.parse_args()

# Copy all files from idir to odir
if os.path.exists(args.odir):
    shutil.rmtree(args.odir)
shutil.copytree(args.idir, args.odir)

# Copy topology line by line, scaling dihedrals defined by dihtype by factor defined by scale
top = []

with open(args.idir + "/" + args.itpfile, "r") as f:  # Read topology file
    in_dih = False
    for line in f:
        if not in_dih:
            top.append(line)
        else:  # process dihedrals
            if line.strip()[0] == ';':  # copy comments as-is
                top.append(line)

            elif args.dihtype == "all" or "-".join(line.strip().split()[:4]) == args.dihtype:  # scale these dihedrals
                orig = line.strip().split()
                mod = orig[:5]
                newvals = ["%.5f" % (args.scale * float(val)) for val in orig[5:]]
                mod.extend(newvals)
                top.append("\t".join(mod) + "\n")

            else:  # copy these dihedrals as-is
                top.append(line)

        if line.strip() == '[ dihedraltypes ]':
            in_dih = True

with open(args.odir + "/" + args.itpfile, "w") as of:  # Open itpfile in odir for writing (overwrites existing file)
    for line in top:
        # write line
        of.write(line)
