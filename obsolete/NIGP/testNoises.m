
addpath ~/Dropbox/PhD/ctrl/gp/NIGP
addpath ~/Dropbox/PhD/ctrl/util

Nls = 1000;

dynmodel0 = trainNIGP(dynmodel, plant, Nls, 0);
%dynmodel1 = trainNIGP(dynmodel, plant, Nls, 1);
dynmodel2 = trainNIGP(dynmodel, plant, Nls, 2);

%exp([dynmodel0.hyp(end,:)' dynmodel1.hyp(end,:)' dynmodel2.hyp(end,:)'])

exp([dynmodel0.hyp(end,:)' dynmodel2.hyp(end,:)'])