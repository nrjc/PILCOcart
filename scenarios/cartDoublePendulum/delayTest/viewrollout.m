close all;
clear all;
setdir;
load CartDoubleSwingup79_H60.mat;
plotall;
animate(latent(j+1), data(j+1), dt, cost);
