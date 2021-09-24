#!/bin/bash
shopt -s extglob

#length of carbon chain
CL=45

#temperature
T=298

#kappa
kappa=0.0243

nstarvals=(-240 -220 -200 -180 -160 -140 -120 -100 -80 -60 -40 -20
0 20 40 60 80 100 120 130 140 150 160 170 180 190 200 220 240 260 280
300 320 340 360 380 400 420 440 460 480 500 520 540 560 580)

cd K$kappa

for nstar in ${nstarvals[@]}; do
    cd K${kappa}N${nstar}
    echo ${nstar}
    # Only do this for C45 group
    rm -- !(ntw*.dat)
    cd ..
done
