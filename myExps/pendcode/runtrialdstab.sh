#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	cp ~/PILCOcart/scenarios/actualExp/bias.txt ~/PILCOcart/scenarios/actualExp/weights.txt ./
	while [ ! -f centers.txt ] || [ ! -f weights.txt ]
	do
	    sleep 50
	    #echo -n '.'
	    cp ~/PILCOcart/scenarios/actualExp/bias.txt ~/PILCOcart/scenarios/actualExp/weights.txt ./
	done
	rm ~/PILCOcart/scenarios/actualExp/*txt
	./autoStab
else
	./autoStab $3
fi


cp state.txt ~/PILCOcart/scenarios/actualExp

rm -R ../benchdata/double/$2
mkdir ../benchdata/double/$2
mv *.txt ../benchdata/double/$2
mv test.avi ../benchdata/double/$2
