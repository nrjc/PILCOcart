#!/bin/sh
matlab_exec=/misc/apps/matlab/matlabR2015b/bin/matlab
batchnum=0
for error in 0.05 0.053 0.055 0.057 0.060
do
for delay in 0.25 1 2 4
do
nohup ${matlab_exec} -nodesktop -nosplash -nodisplay -r "try;doitSwingup($error,$delay,$batchnum);exit;" > nohup${batchnum}.out 2>&1&
batchnum=$((batchnum+1))
done
done