#!/bin/bash

#length of carbon chain
CL=45

#temperature
T=298

#kappa
kappa=0.0243

#Nstar values
nstarvals=(-240 -220 -200 -180 -160 -140 -120 -100 -80 -60 -40 -20
0 20 40 60 80 100 120 130 140 150 160 170 180 190 200 220 240 260 280
300 320 340 360 380 400 420 440 460 480 500 520 540 560 580)

mkdir -p K$kappa
cd K$kappa

for nstar in ${nstarvals[@]}; do
    mkdir -p K${kappa}N${nstar}
    cd K${kappa}N${nstar}

    cp ../../run_OP_template.sh run_OP.sh
    sed -i "s/XXNSTARXX/$nstar/g" run_OP.sh
    sed -i "s/XXLENGTHXX/$CL/g" run_OP.sh
    sed -i "s/XXTEMPXX/$T/g" run_OP.sh
    sed -i "s/XXKAPPAXX/$kappa/g" run_OP.sh
    sbatch run_OP.sh
    cd ..
done
