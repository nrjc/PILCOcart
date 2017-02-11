#!/bin/bash

./runtrialDMK3L.sh 1 trialDMK3j0L

for a in {1..20}
do
  ./runtrialDMK3L.sh 0 trialDMK3j${a}L
  echo -n "Trial $a"
done
