#!/bin/sh
matlab_exec=/misc/apps/matlab/matlabR2015b/bin/matlab
batchnum=20
nohup ${matlab_exec} -nodesktop -nosplash -nodisplay -r "runStabilizationDelay;exit;" > nohup${batchnum}.out 2>&1&
