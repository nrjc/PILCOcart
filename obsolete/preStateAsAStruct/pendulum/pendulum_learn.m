% Pendulum experiment: Gaussian-RBF policy
%
% Copyright (C) 2008-2012 by Marc Deisenroth & Carl Edward Rasmussen,
% 2012-07-10. Edited by Joe Hall 2012-10-02. 
% Edited by Jonas Umlauft 2014-03-31
clear all; close all;

settings;
basename = 'pendulum_';

%% Initial Rollouts (apply random actions)
for j = 1:J                                   % get the first observations
  [data(j), latent{j}, realCost{j}] = rollout(gaussian(mu0, S0), ...
    struct('policy',struct('maxU',ctrl.policy.maxU),'U',ctrl.U), H, plant, cost);
  disp([data(j).state [data(j).action; zeros(1,ctrl.U)]]); disp(' ');
  draw_rollout;
end


mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

%% Start model-based policy search
for j =  1:N
  trainDynModel;
  learnPolicy;  
  applyController;
  draw_rollout; distFig();
  disp(['controlled trial # ' num2str(j)]);
end