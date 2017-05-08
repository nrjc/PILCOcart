close all;
clear all;
setdir;

load ./CartMixedTraining38_H30.mat
plotall;
currentrun = J+j;
animate(latent(currentrun), data(currentrun), dt, cost);
% dyn.induce = zeros(0,12,6);
% dyn.inputs=dyn.inputs(ceil(length(dyn.inputs)/2):end,:);
% dyn.target = dyn.target(ceil(length(dyn.target)/2):end,:);
% dyn.inputs = [dyn.inputs;dyn2.inputs];
% dyn.target = [dyn.target;dyn2.target];
% dyn.trainmanual(dyni,plant.dyno);
% load ./CartDoubleSwingupRestart50_H40.mat ctrl
% mu0=[0;0;0;0;0;pi;pi];
% s.m=mu0;
% H=15;
% learnPolicy;
% H=30;
% learnPolicy;
% H=35;
% learnPolicy;
% H=40;
% learnPolicy;