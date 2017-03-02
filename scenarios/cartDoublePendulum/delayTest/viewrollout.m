close all;
clear all;
setdir;
load ./swingup/CartDoubleSwingupRestart93_H60.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);