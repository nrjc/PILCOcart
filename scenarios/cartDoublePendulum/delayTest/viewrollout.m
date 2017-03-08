close all;
clear all;
setdir;
load ./swingup/CartDoubleSwingupFixedWeight100_H60.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);