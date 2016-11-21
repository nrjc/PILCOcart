#!/bin/bash
#
# To call:  ./makeall.sh
# (when in the same directory as this file)
file='Makefile'
while read line; do
  var=$(echo $line | grep -o "^.*:" | grep -o ".*[^:]")
  if [ -n "$var" ]; then
    make $var
  fi
done < $file
