#!/bin/bash

for a in {6..20}
do
  ./runtrial.sh 0 trialzzz${a}
  echo -n "Trial $a"
done
