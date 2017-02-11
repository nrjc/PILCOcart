for a in {6..150}
do
  ./runtrialDMK3L.sh 0 trialDMK3f${a}L
  echo -n "Trial $a"
done
