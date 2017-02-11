clear all;
setdir;
load CartDoubleSwingup40_H60.mat;
plotall;
animate(latent(41), data(41), dt, cost);
