close all;
clear all;
setdir;
load ./swingup/CartDoubleSwingupLinearSeqShort3p99_H30.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);