close all;
clear all;
setdir;
load ./swingup/CartDoubleSwingupLinearSeqShort32_H10.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);