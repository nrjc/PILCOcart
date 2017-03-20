close all;
clear all;
setdir;
load ./CartDoubleSwingupRestart72_H30.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);