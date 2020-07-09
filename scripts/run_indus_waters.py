"""Runs waters analysis by creating, processing, and calling a WatersAnalysis
object. Arguments are read from the command line."""

from INDUSAnalysis import indus_waters
from INDUSAnalysis.lib import profiling
import matplotlib.pyplot as plt


@profiling.timefunc
def main():
    warnings = ""
    waters = indus_waters.WatersAnalysis()
    waters.parse_args()
    waters.read_args()
    startup_string = "#### INDUS Waters ####\n" + warnings + "\n"
    print(startup_string)
    waters()
    plt.close('all')


if __name__ == "__main__":
    main()
