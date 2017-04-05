close all;
clear all;
setdir;
load combinedDyn
dyn.inf_method='full';
dyn.induce=zeros(0,6,12);
dyn.trainmanual(dyni,plant.dyno);
learnPolicy;
basename='CartCombinedDyn';
applyController;
