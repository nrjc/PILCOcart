% Script to perform the policy searches and apply the learned policy
%
% (C) Copyright 2010-2012 by Carl Edward Rasmussen & Marc Deisenroth 2012-03-16

Du = length(plant.maxU); Da = length(plant.angi);   % number of ctrl and angles
xaug = [x(:,dyno) x(:,end-Du-2*Da+1:end-Du)];         % x augmented with angles
dynmodel.inputs = [xaug(:,dyni) x(:,end-Du+1:end)]; % use dyni and  ctrl inputs
dynmodel.target = y(:,dyno);
dynmodel.target(:,difi) = dynmodel.target(:,difi) - x(:,dyno(difi));    
dynmodel = dynmodel.train(dynmodel, plant, [50 50], 0);   % train the dyn model
Xh = dynmodel.hyp;
disp(['learned noises: ' num2str(exp(Xh(end,:)))]);
disp(['SNRs          : ' num2str(exp(Xh(end-1,:)-Xh(end,:)))]);

plant.seed = plant.seed + 1;

figure(1);                                                  % update the policy
[policy fX3] = minimize(policy, 'partValue', struct('length', -150, ...
                     'verbosity', 3), mu0Sim, S0Sim, dynmodel, plant, cost, H);
figure(2); hold on; plot(fX3); drawnow;

% predict rollout from sampled start state
M{j} = partPred(policy, plant, dynmodel, mu0Sim(:,1), S0Sim, H);

[fantasy.mean{j} fantasy.std{j}] = calcPartCost(cost, M{j});         % find cost    

figure(3); errorbar(fantasy.mean{j},2*fantasy.std{j});

if H < maxH && fantasy.mean{j}(end) < 1/2                   % increase horizon?
  H = min(maxH,ceil(1.25*H)); disp(['new horizon: ' num2str(H) ' steps']);
end

if isfield(plant,'constraint'), HH = maxH; else HH = H; end        % do rollout
[xx, yy, realCost{j}, latent{j+J}] = ...
                           rollout(gaussian(mu0, S0), policy, HH, plant, cost);
disp(xx);                                                  % display new states
x = [x; xx]; y = [y; yy];                                % augment training set

filename = [basename num2str(j) '_H' num2str(H)]; save(filename);   % save data
