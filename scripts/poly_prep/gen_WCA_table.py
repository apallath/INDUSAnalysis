"""
Generates tabulated WCA-scaling potential for use with GROMACS 4+.
Uses Lorentz-Berthelot combination rules for estimating sigma and epsilon.
Inspired by https://github.com/vasudevanv/pots.

WCA-scaling potential:

        4*\eps*[ (sigma/r)**12 - (sigma/r)**6 + (1 - \lambda)/4 ] , r< 2**(1/6) \sigma
u(r) =
        4*\eps*\lambda*[ (sigma/r)**12 - (sigma/r)**6 ] , r >= 2**(1/6) \sigma


Setting \lambda = 0 gives you the WCA potential
Setting \lambda = 1 gives you the full Lennard-Jones potential
Intermediate values of \lambda scale down the attractions

Setting noscale = True modifies 'g' term instead of 'h'
"""
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("group1", help="Name of first group")
parser.add_argument("group2", help="Name of second group")
parser.add_argument("sigma1", type=float, help="LJ sigma parameter for first group")
parser.add_argument("epsilon1", type=float, help="LJ epsilon parameter for first group")
parser.add_argument("sigma2", type=float, help="LJ sigma parameter for second group")
parser.add_argument("epsilon2", type=float, help="LJ epsilon parameter for second group")

parser.add_argument("-scale", type=float, default=0, help="Scaling factor (lambda = 0 => WCA, lambda = 1 => LJ, any values in between => scaled LJ, default = 0)")
parser.add_argument("-rcut", type=float, default=1, help="van der Waals cutoff radius (default = 1 nm)")
parser.add_argument("-rext", type=float, default=1, help="Generate tables until rcut + rext (default = 1 nm)")
parser.add_argument("-rspace", type=float, default=0.002, help="Spacing between r values (default = 0.002 nm)")
parser.add_argument("-mixingrule", type=int, default=2, help="LJ mixing rule (allowed options: 2 = Lorentz-Berthelot mixing rules, 3 = geometric average for both parameters, default = 2)")

parser.add_argument("--noscale", action="store_true", help="Scale attractive interactions")
parser.add_argument("--nocoulomb", action="store_true", help="Set the charge-related columns to 0")

args = parser.parse_args()

# Table Parameters
rmax = args.rcut + args.rext
bmax = int(rmax / args.rspace)

# Create the table
fname = "table_" + args.group1.strip() + "_" + args.group2.strip() + ".xvg"

# Mixing
if args.mixingrule == 2:
    sigma = (args.sigma1 + args.sigma2) / 2.0
    eps =  math.sqrt(args.epsilon1 * args.epsilon2)
elif args.mixingrule == 3:
    sigma = math.sqrt(args.sigma1 * args.sigma2)
    eps =  math.sqrt(args.epsilon1 * args.epsilon2)
else:
    raise ValueError("Invalid mixing rule")

with open(fname, 'w') as file:
    # Set output format
    fmt = "%14.10e\t" * 7 + "\n"

    # Generate the table
    for i in range(bmax + 1):
        x = i * args.rspace

        if (x < 4e-2) :
            file.write(fmt % (x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        elif (x >= 4e-2 and x <= 2 ** (1.0 / 6.0) * sigma) :
            if not args.nocoulomb:
                f = 1.0 / x
                fprime = -1.0 / x ** 2
            else:
                f = 0
                fprime = 0

            if not args.noscale:
                g = -1.0 / x ** 6.
                gprime = 6.0 / x ** 7
                h = 1.0 / x ** 12 + (1 - args.scale) / (4.0 * sigma ** 12)
                hprime = -12.0 / x ** 13
            else:
                g = -1.0 / x ** 6 + 1 / (4.0 * sigma ** 6)
                gprime = 6.0 / x ** 7
                h = 1.0 / x ** 12
                hprime = -12.0 / x ** 13

            file.write(fmt % (x, f, -fprime, g, -gprime, h, -hprime))

        else:
            if not args.nocoulomb:
                f = 1.0 / x
                fprime = -1.0 / x ** 2
            else:
                f = 0
                fprime = 0

            if not args.noscale:
                g = -args.scale / x ** 6
                gprime = 6.0 * args.scale / x ** 7
                h = args.scale * 1.0 / x ** 12
                hprime = -12.0 * args.scale * 1.0 / x ** 13
            else:
                g = 0
                gprime = 0
                h = 0
                hprime = 0

            file.write(fmt % (x, f, -fprime, g, -gprime, h, -hprime))
