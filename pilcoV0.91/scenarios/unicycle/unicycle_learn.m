%% unicycle_learn.m
% *Summary:* Script to learn a controller for the unicycle balancing
%
% Copyright (C) 2008-2014 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2014-02-13
%
%% High-Level Steps
% # Load parameters
% # Create J initial trajectories by applying random controls
% # Controlled learning (train dynamics model, policy learning, policy
% application)

%% Code
% 1. Initialization
clear all; close all;
settings_unicycle;                % load scenario-specific settings
basename = 'unicycle_';           % filename used for saving data

% 2. Initial J random rollouts
for jj = 1:J
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
  x = [x; xx]; y = [y; yy];       % augment training sets for dynamics model
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
    draw(latent(jj),plant,1/25,cost,J);
  end
  
end

% mu0, S0:       for interaction
% mu0Sim, S0Sim: for internal simulation (partial state only)
mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

% 3. Controlled learning (N iterations)
for j = 1:N
  trainDynModel;   % train (GP) dynamics model
  learnPolicy;     % learn policy
  applyController; % apply controller to system
  disp(['controlled trial # ' num2str(j)]);
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
    draw(latent(jj),plant,1/25,cost,J);
  end
end