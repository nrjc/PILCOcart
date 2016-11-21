% Script to perform the policy searches
%
% (C) Copyright 2010-2012 by Carl Edward Rasmussen, Marc Deisenroth and
%       Andrew McHutchon, 2014-04-03

% 1. Update the policy
opt.fh = 1;
[ctrl.policy.p, fX3] = minimize(ctrl.policy.p, 'value', opt, mu0Sim, S0Sim, dynmodel, ctrl, plant, cost, H);

if ~ishandle(2); figure(2); else set(0,'CurrentFigure',2); end
hold on; plot(fX3); drawnow;

% 2. Predict rollout from sampled start state and find cost
[M{j}, Sigma{j}] = pred(ctrl, plant, dynmodel, mu0Sim(:,1), S0Sim, H); 
[fantasy.mean{j}, fantasy.std{j}] = calcCost(cost, M{j}, Sigma{j}); % predict cost trajectory

if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end
clf(3); errorbar(0:H,fantasy.mean{j},2*fantasy.std{j}); drawnow;
