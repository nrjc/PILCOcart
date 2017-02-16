close all;
clear all;
setdir;
load CartDoubleSwingup74_H60.mat;
plotall;
animate(latent(74), data(74), dt, cost);
