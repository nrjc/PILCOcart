#!/bin/bash

for a in {1..20}
do
  ./runtrialMK.sh 0 trialMKs${a}
  echo -n "Trial $a"
done
