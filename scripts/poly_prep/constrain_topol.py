"""
Adds atom-atom distance constraints (and removes pairwise interactions) in GMX topology file.
"""
import argparse
import MDAnalysis as mda

parser = argparse.ArgumentParser()
parser.add_argument("ifile", help="Topology file")
parser.add_argument("-ofile", help="Modified topolgy file (default=topol_constr.top)", default="topol_constr.top")

args = parser.parse_args()

# Topology line by line
top = []
bonds = []
constr = []

# Read topology file
with open(args.ifile, "r") as f:
    in_bonds = False
    in_pairs = True
    for line in f:
        top.append(line)
        if line.strip() == '[ bonds ]':
            in_bonds = True
        elif line.strip() == '[ pairs ]':
            in_bonds = False
        elif in_bonds:
            if len(line.strip()) >= 1:
                if line.strip()[0] != ';':
                    bonds.append(line)

constr.append("#ifndef FLEXIBLE\n\n[ constraints ]\n")

# Process constraints
for bond in bonds:
    constr.append(bond.strip() + "\t0.154\n")

constr.append("\n#else\n\n")

# Write topology file:
with open(args.ofile, "w") as of:
    for line in top:
        if line.strip() == '[ bonds ]':
            # Write constraints first
            for con in constr:
                of.write(con)
            # write line
            of.write(line)

        elif line.strip() == '[ pairs ]':
            of.write("#endif\n\n")
            of.write(line)

        else:
            of.write(line)
