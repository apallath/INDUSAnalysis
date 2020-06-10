from analysis.protein_order_params import OrderParams
from meta_analysis.profiling import timefunc

@timefunc
def main():
    warnings = "Proceed with caution: this script requires PBC-corrected protein structures!"
    prot = OrderParams()
    prot.parse_args()
    prot.read_args()
    startup_string = "#### Order Parameter Analysis ####\n" + warnings + "\n"
    print(startup_string)
    prot()
    plt.close('all')

if __name__=="__main__":
    main()
