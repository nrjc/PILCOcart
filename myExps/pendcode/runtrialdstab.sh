#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	scp cbl-fs:"~/PILCOcart/scenarios/actualExp/bias.txt ~/PILCOcart/scenarios/actualExp/weights.txt" ./
	while [ ! -f centers.txt ] || [ ! -f weights.txt ]
	do
	    sleep 50
	    #echo -n '.'
	    scp cbl-fs:"~/PILCOcart/scenarios/actualExp/bias.txt ~/PILCOcart/scenarios/actualExp/weights.txt" ./
	done
	ssh cbl-fs 'rm ~/PILCOcart/scenarios/actualExp/*txt'
fi

./autodoubleStab
scp state.txt cbl-fs:~/PILCOcart/scenarios/actualExp

rm -R ../benchdata/double/$2
mkdir ../benchdata/double/$2
mv *.txt ../benchdata/double/$2
mv test.avi ../benchdata/double/$2
