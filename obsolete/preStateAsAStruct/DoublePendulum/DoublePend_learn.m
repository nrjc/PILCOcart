% Pendubot experiment: Gaussian-RBF policy
%
% Copyright (C) 2008-2012 by Marc Deisenroth and Carl Edward Rasmussen,
% 2012-01-19. Edited by Joe Hall 2012-10-02, Edited by Jonas Umlauft 2014-06-30
clear all; close all;

settings;
basename = 'doublepend_';

%% Initial Rollouts (apply random actions)
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);
for j = 1:J                                   % get the first observations
  [data(j), latent{j}, realCost{j}] = rollout(gaussian(mu0, S0), ...
                               struct('policy',struct('maxU',ctrl.policy.maxU),'U',ctrl.U), H, plant, cost);
  disp([data(j).state [data(j).action;zeros(1,ctrl.U)]]);

  figure(5); clf; %draw_rollout;
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
