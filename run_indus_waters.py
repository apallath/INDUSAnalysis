from analysis.indus_waters import IndusWaters
from meta_analysis.profiling import timefunc

@timefunc
def main():
    warnings = ""
    waters = IndusWaters()
    waters.parse_args()
    waters.read_args()
    startup_string = "#### INDUS Waters ####\n" + warnings + "\n"
    print(startup_string)
    waters()
    plt.close('all')

if __name__=="__main__":
    main()
