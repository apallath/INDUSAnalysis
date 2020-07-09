"""Runs contacts analysis by creating, processing, and calling a ContactsAnalysis
object. Arguments are read from the command line."""

from INDUSAnalysis import contacts
from INDUSAnalysis.lib import profiling
import matplotlib.pyplot as plt


@profiling.timefunc
def main():
    warnings = "Proceed with caution: this script requires PBC-corrected protein structures!\n"
    cts = contacts.ContactsAnalysis()
    cts.parse_args()
    cts.read_args()
    startup_string = "#### Contacts ####\n" + warnings
    print(startup_string)
    cts()
    plt.close('all')


if __name__ == "__main__":
    main()
