#!/bin/bash

./runtrialDMK3.sh 1 trialDMK3e0

for a in {1..20}
do
  ./runtrialDMK3.sh 0 trialDMK3e${a}
  echo -n "Trial $a"
done
