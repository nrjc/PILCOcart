#!/bin/sh
matlab_exec=/misc/apps/matlab/matlabR2015b/bin/matlab
batchnum=22
nohup ${matlab_exec} -nodesktop -nosplash -nodisplay -r "doitSwingupContinue;exit;" > nohup${batchnum}.out 2>&1&
