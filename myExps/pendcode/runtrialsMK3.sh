#!/bin/bash

./runtrialMK3.sh 1 trialMK3b0

for a in {1..20}
do
  ./runtrialMK3.sh 0 trialMK3b${a}
  echo -n "Trial $a"
done
