"""Runs protein order parameters analysis by creating, processing, and calling an
OrderParamsAnalysis object. Arguments are read from the command line."""

from INDUSAnalysis import protein_order_params
from INDUSAnalysis.lib import profiling
import matplotlib.pyplot as plt


@profiling.timefunc
def main():
    warnings = "Proceed with caution: this script requires PBC-corrected protein structures!"
    prot = protein_order_params.OrderParamsAnalysis()
    prot.parse_args()
    prot.read_args()
    startup_string = "#### Order Parameter Analysis ####\n" + warnings + "\n"
    print(startup_string)
    prot()
    plt.close('all')


if __name__ == "__main__":
    main()
