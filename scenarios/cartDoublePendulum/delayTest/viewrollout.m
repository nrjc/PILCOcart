close all;
clear all;
setdir;
load ./CartDoubleSwingupRestart90_H60.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);