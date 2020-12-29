"""
Generate umbrellas for dynamic union of spherical shells

TODO: Implement for wide variety of umbrella configurations.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ifname", help="index file")
parser.add_argument("-o", help="output file [default=umbrella_template.conf]")
parser.add_argument("-start", help="shell start (nm) [default=-1]")
parser.add_argument("-end", help="shell end (nm) [default=0.6]")

args = parser.parse_args()

ifn = args.ifname

indices = []

with open(ifn, "r") as ifile:
    for line in ifile:
        indices = indices + [int(i) for i in line.strip().split()]

ofn = args.o
if ofn is None:
    ofn = "umbrella_template.conf"

ofn_header = """; Umbrella potential for a spherical shell cavity
; Name      Type                Group   Kappa   Nstar   mu      width   cutoff  outfile      nstout
hydshell    dyn_union_sph_sh    OW      0.0     0       PHIXX   0.01    0.02    phioutPHIXX.dat   50 \\"""

start = args.start
if start is None:
    start = -1

end = args.end
if end is None:
    end = 0.6

with open(ofn, "w") as ofile:
    ofile.write(ofn_header + "\n")
    for idx in indices[:-1]:
        ofile.write("{}    {}    {} \\\n".format(start, end, idx))
    ofile.write("{}    {}    {}\n".format(start, end, indices[-1]))
