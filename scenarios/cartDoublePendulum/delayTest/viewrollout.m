close all;
clear all;
setdir;
load ./swingup/CartDoubleSwingupLinearSeqShort2p27_H20.mat
plotall;
currentrun = j+1;
animate(latent(currentrun), data(currentrun), dt, cost);