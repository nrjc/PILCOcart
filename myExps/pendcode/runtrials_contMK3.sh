#!/bin/bash

for a in {1..20}
do
  ./runtrialMK3.sh 0 trialMK3b${a}
  echo -n "Trial $a"
done
