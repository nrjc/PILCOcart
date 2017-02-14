close all;
clear all;
setdir;
load CartDoubleSwingup98_H60.mat;
plotall;
animate(latent(99), data(99), dt, cost);
