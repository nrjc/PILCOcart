#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	cp ~/PILCOcart/scenarios/actualexp/bias.txt ~/PILCOcart/scenarios/actualexp/weights.txt ./
	while [ ! -f centers.txt ] || [ ! -f weights.txt ]
	do
	    sleep 50
	    #echo -n '.'
	    cp ~/PILCOcart/scenarios/actualexp/bias.txt ~/PILCOcart/scenarios/actualexp/weights.txt ./
	done
	rm ~/PILCOcart/scenarios/actualexp/*txt
	read
	./autoStab
else
	read
	./autoStab $3
fi


cp state*.txt ~/PILCOcart/scenarios/actualexp

rm -R ../benchdata/double/$2
mkdir ../benchdata/double/$2
mv *.txt ../benchdata/double/$2
mv test.avi ../benchdata/double/$2
