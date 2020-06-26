from INDUSAnalysis.contacts import Contacts
from INDUSAnalysis.lib.profiling import timefunc
import matplotlib.pyplot as plt

@timefunc
def main():
    warnings = "Proceed with caution: this script requires PBC-corrected protein structures!\n"
    contacts = Contacts()
    contacts.parse_args()
    contacts.read_args()
    startup_string = "#### Contacts ####\n" + warnings
    print(startup_string)
    contacts()
    plt.close('all')

if __name__=="__main__":
    main()
