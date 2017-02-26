close all;
clear all;
setdir;
load ./enumeratedelay/CartDoubleSwingup40_H65.mat
plotall;
animate(latent(j+1), data(j+1), dt, cost);
