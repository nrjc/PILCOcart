close all;
clear all;
setdir;
load ./swingup/CartDoubleSwingupRestart61_H60.mat
plotall;
animate(latent(j+1), data(j+1), dt, cost);
