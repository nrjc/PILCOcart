%% trainDynModel.m
% *Summary:* Script to learn the dynamics model
%
% Copyright (C) 2008-2014 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modification: 2014-02-03
%
%% High-Level Steps
% # Extract states and controls from x-matrix
% # Define the training inputs and targets of the GP
% # Train the GP 

%% Code

% 1. Train GP dynamics model
Du = length(policy.maxU); Da = length(plant.angi);% no. of ctrl and angles
xaug = [x(:,dyno) x(:,end-Du-2*Da+1:end-Du)];     % x augmented with angles
dynmodel.inputs = [xaug(:,dyni) x(:,end-Du+1:end)];    % use dyni and ctrl
dynmodel.targets = y(:,dyno);

dynmodel = dynmodel.train(dynmodel, trainOpt);  % train dynamics GP

% display some hyperparameters
h = dynmodel.hyp;                               % display hyperparameters
disptable([exp([h.n]); exp([h.on]); sqrt(dynmodel.noise); exp([h.s])./sqrt(dynmodel.noise)],...
    varNames,'process noises: |observ. noises: |total noises: |SNRs: ')

% signal-to-noise ratios (values > 500 can cause numerical problems)