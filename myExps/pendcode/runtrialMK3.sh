#!/bin/bash

# rm *txt
if [ $1 = "0" ]; then
	cp ~/PILCOcart/scenarios/cartPole/centers.txt ~/PILCOcart/scenarios/cartPole/weights.txt ~/PILCOcart/scenarios/cartPole/W.txt ./
	while [ ! -f centers.txt ] || [ ! -f weights.txt ] || [ ! -f W.txt ]
	do
	    sleep 50
	    #echo -n '.'
		cp ~/PILCOcart/scenarios/cartPole/centers.txt ~/PILCOcart/scenarios/cartPole/weights.txt ~/PILCOcart/scenarios/cartPole/W.txt ./
	done
	rm ~/PILCOcart/scenarios/cartPole/*txt
fi

./autosingleMK3
cp state.txt ~/PILCOcart/scenarios/cartPole/

rm -R ~/benchdata/single/$2
mkdir ~/benchdata/single/$2
mv *.txt ~/benchdata/single/$2
mv test.avi ~/benchdata/single/$2
