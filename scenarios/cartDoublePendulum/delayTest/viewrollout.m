clear all;
setdir;
load CartDoubleSwingup15_H30.mat;
plotall;
animate(latent(16), data(16), dt, cost);
