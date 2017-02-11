#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	scp cbl-fs:"~/ctrl/scenarios/cartDoublePendulum/weights.txt ~/ctrl/scenarios/cartDoublePendulum/biases.txt" ./
	while [ ! -f weights.txt ] || [ ! -f biases.txt ]
	do
	    sleep 50
	    #echo -n '.'
	    scp cbl-fs:"~/ctrl/scenarios/cartDoublePendulum/weights.txt ~/ctrl/scenarios/cartDoublePendulum/biases.txt" ./
	done
	ssh cbl-fs 'rm ~/ctrl/scenarios/cartDoublePendulum/*txt'
fi

./autodoubleMK3L
scp state.txt cbl-fs:~/ctrl/scenarios/cartDoublePendulum/

rm -R ../benchdata/double/$2
mkdir ../benchdata/double/$2
mv *.txt ../benchdata/double/$2
mv test.avi ../benchdata/double/$2
