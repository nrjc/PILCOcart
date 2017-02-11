#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	scp cbl-fs:"~/ctrl/scenarios/cartPole/centers.txt ~/ctrl/scenarios/cartPole/weights.txt ~/ctrl/scenarios/cartPole/W.txt" ./
	while [ ! -f centers.txt ] || [ ! -f weights.txt ] || [ ! -f W.txt ]
	do
	    sleep 50
	    #echo -n '.'
	    scp cbl-fs:"~/ctrl/scenarios/cartPole/centers.txt ~/ctrl/scenarios/cartPole/weights.txt ~/ctrl/scenarios/cartPole/W.txt" ./
	done
	ssh cbl-fs 'rm ~/ctrl/scenarios/cartPole/*txt'
fi

./autosingleMK
scp state.txt cbl-fs:~/ctrl/scenarios/cartPole/

rm -R ../benchdata/single/$2
mkdir ../benchdata/single/$2
mv *.txt ../benchdata/single/$2
mv test.avi ../benchdata/single/$2
