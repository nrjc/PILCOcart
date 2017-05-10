#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	scp cbl-fs:"~/ctrl/scenarios/cartDoublePendulum/bias.txt ~/ctrl/scenarios/cartDoublePendulum/weights.txt" ./
	while [ ! -f centers.txt ] || [ ! -f weights.txt ]
	do
	    sleep 50
	    #echo -n '.'
	    scp cbl-fs:"~/ctrl/scenarios/cartDoublePendulum/bias.txt ~/ctrl/scenarios/cartDoublePendulum/weights.txt" ./
	done
	ssh cbl-fs 'rm ~/ctrl/scenarios/cartDoublePendulum/*txt'
fi

./autodoubleStab
scp state.txt cbl-fs:~/ctrl/scenarios/cartDoublePendulum/

rm -R ../benchdata/double/$2
mkdir ../benchdata/double/$2
mv *.txt ../benchdata/double/$2
mv test.avi ../benchdata/double/$2
