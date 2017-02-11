#!/bin/bash

#cp ../benchdata/myrandom/centers.txt ./
./runtrial.sh 1 trialzzzz0

for a in {1..20}
do
  ./runtrial.sh 0 trialzzzz${a}
  echo -n "Trial $a"
done
