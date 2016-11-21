% Arm movement experiment

clear all; close all;

settingsff_carl;
plant.noise = (0.2 * diag([1 1 1/dt 1/dt])).^2;
odeFIELD = 1;


checkSettings;

%% Initial Rollouts (apply random actions)
for j = 1:J                                   % get the first observations
  [data(j), latent{j}, realCost{j}] = rollout(gaussian(mu0, S0), ...
                              struct('policy',struct('maxU',ctrl.policy.maxU),'U',ctrl.U,'init',[]), H, plant, cost);
  disp([data(j).state [data(j).action;zeros(1,length(ctrl.policy.maxU))]]);
end

% mu0, S0:       for interaction
% mu0Sim, S0Sim: for internal simulation (partial state only)
mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

%% Start model-based policy search
tic
for j = 1:N
  trainDynModel;
  learnPolicy;  
  applyController;
  drawRollouts;
  disp(['controlled trial # ' num2str(j)]);
  
  % Save cost and state plots for each trial (png and fig)
  h = get(0,'children');
  for i=1:3
    saveas(h(i), [basename num2str(j) '_figure' num2str(length(h) + 1 - i)], 'fig');
  end
end

% Save minimisation graph
saveas(h(4), [basename num2str(j) '_figure' num2str(length(h) + 1 - 4)], 'fig');
toc