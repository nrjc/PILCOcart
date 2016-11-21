% Pendubot experiment: Gaussian-RBF policy
%
% Copyright (C) 2008-2012 by Marc Deisenroth and Carl Edward Rasmussen,
% 2012-01-19. Edited by Joe Hall 2012-10-02.
clear all; close all;

settings;
basename = 'pendubot_';

%% Initial Rollouts (apply random actions)
for j = 1:J                                   % get the first observations
  [xx, yy, realCost{j}, latent{j}] = rollout(gaussian(mu0, S0), ...
                              struct('maxU',policy.maxU), H, plant, cost);
  xx
  x = [x; xx]; y = [y; yy];
  figure(5); clf; draw_rollout;
end

% mu0, S0:       for interaction
% mu0Sim, S0Sim: for internal simulation (partial state only)
mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

%% Start model-based policy search
for j = 1:N
  trainDynModel;
  learnPolicy;  
  applyController;
  if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
  draw_rollout;
  disp(['controlled trial # ' num2str(j)]);
end