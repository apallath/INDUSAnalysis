"""
Script to generate initial config file for INDUSAnalysis.ensemble.polymers.wham_nt_ntrg
"""
import yaml

NBEAD = 45
TEMP = 298

wd_nt = "inputs_ensemble.polymer.wham_nt_ntrg/C{}_T{}".format(NBEAD, TEMP)
wd_rg = "inputs_ensemble.polymer.wham_nt_ntrg/C{}_T{}_OP".format(NBEAD, TEMP)

TSTEP = 0.002
UMBDT = 50
XTCDT = 100

KAPPA = 0.0243
nstars = list(range(-240, 580 + 1, 20))

windows = {}
windows["unbiased"] = {"Nt_file": wd_nt + '/unbiased/nt_ntw_k0_N0.dat',
                       "Rg_file": wd_rg + '/unbiased/C{}_Rg.pkl'.format(NBEAD),
                       "TSTEP": TSTEP,
                       "UMBDT": UMBDT,
                       "XTCDT": XTCDT}
for NSTAR in nstars:
    windows[NSTAR] = {"Nt_file": wd_nt + '/K{kappa}/K{kappa}N{nstar}/ntw_k{kappa}_N{nstar}.dat'.format(kappa=KAPPA, nstar=NSTAR),
                      "Rg_file": wd_rg + '/K{kappa}/K{kappa}N{nstar}/C{nbead}_Rg.pkl'.format(kappa=KAPPA, nstar=NSTAR, nbead=NBEAD),
                      "TSTEP": TSTEP,
                      "UMBDT": UMBDT,
                      "XTCDT": XTCDT}

# Config dictionary
with open(r'config_template.yaml', 'r') as th:
    config = yaml.safe_load(th)

config["system"]["NBEAD"] = NBEAD
config["system"]["TEMP"] = TEMP
config["umbrellas"]["KAPPA"] = KAPPA
config["windows"] = windows

# Write config to file
with open(r'config.yaml', 'w') as fh:
    yaml.dump(config, fh)
