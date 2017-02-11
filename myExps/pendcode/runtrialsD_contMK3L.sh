#!/bin/bash

for a in {144..200}
do
  ./runtrialDMK3L.sh 0 trialDMK3h${a}L
  echo -n "Trial $a"
done
