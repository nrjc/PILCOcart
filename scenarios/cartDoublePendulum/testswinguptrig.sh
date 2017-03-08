#!/bin/sh
matlab_exec=/misc/apps/matlab/matlabR2015b/bin/matlab
batchnum=4
nohup ${matlab_exec} -nodesktop -nosplash -nodisplay -r "doitSwingupTrig;exit;" > nohup${batchnum}.out 2>&1&
