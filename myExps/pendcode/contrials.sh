#!/bin/bash
a=1
while true
do
    while [ ! -f start.txt ] ;
    do
	sleep 2
    done
sh ./runtrial.sh "$a"

((a++))
done




