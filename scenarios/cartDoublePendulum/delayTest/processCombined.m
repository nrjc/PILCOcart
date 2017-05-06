close all;
clear all;
setdir;
load combinedDyn
%dyn.inf_method='full';
dyn.induce = [dyn.induce;zeros(800, 12, E)];
dyn.trainmanual(dyni,plant.dyno);
learnPolicy;
basename='CartCombinedDyna';
applyController;
