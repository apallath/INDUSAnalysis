# @author Nick Rego
# Adapted by Akash Pallath.

import argparse

import MDAnalysis
from scipy.spatial import Delaunay


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Remove internal waters from solvated protein')
    parser.add_argument('-c', '--struct', type=str, required=True,
                        help='GRO structure file')
    parser.add_argument('-o', '--out', type=str, default='prot_removed.gro',
                        help='Output GRO structure file (default: %(default)s)')
    parser.add_argument('-b', '--bak', type=str, default='prot_to_remove.gro',
                        help='GRO structure file with waters within convex hull marked in the tempfactor column (default: %(default)s)')
    parser.add_argument('--sel-spec', type=str, default='protein',
                        help='Selection spec for selecting all protein atoms (including hydrogens) \
                              default: %(default)s')

    args = parser.parse_args()

    univ = MDAnalysis.Universe(args.struct)
    prot = univ.select_atoms(args.sel_spec)

    waters = univ.select_atoms('name OW and not ({})'.format(args.sel_spec))

    hull = Delaunay(prot.positions)

    # Positions of waters close to protein (or in it)
    inside_mask = hull.find_simplex(waters.positions) > -1

    print("Removing {} waters".format(inside_mask.sum()))

    univ.add_TopologyAttr('tempfactors')
    waters_to_remove = waters[inside_mask]

    for water in waters_to_remove:
        water.residue.atoms.tempfactors = -1

    univ.write(args.bak)

    atoms = univ.atoms[univ.atoms.tempfactors > -1]
    atoms.write(args.out)

    n_waters = atoms.select_atoms('name OW and not ({})'.format(args.sel_spec)).n_atoms

    print('New n waters: {}'.format(n_waters))
