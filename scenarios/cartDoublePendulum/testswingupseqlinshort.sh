#!/bin/sh
matlab_exec=/misc/apps/matlab/matlabR2015b/bin/matlab
batchnum=7
nohup ${matlab_exec} -nodesktop -nosplash -nodisplay -r "doitSwingupLinearSeqShort3;exit;" > nohup${batchnum}.out 2>&1&
