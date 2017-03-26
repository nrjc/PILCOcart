close all;
clear all;
setdir;
load ./CartDoubleSwingupRestart13_H40.mat
plotall;
currentrun = J+j;
animate(latent(currentrun), data(currentrun), dt, cost);