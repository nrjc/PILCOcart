#!/bin/bash

for a in {38..40}
do
  ./runtrialDMK3.sh 0 trialDMK3e${a}
  echo -n "Trial $a"
done
