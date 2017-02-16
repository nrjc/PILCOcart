% Script to perform the policy searches
%
% (C) Copyright 2010-2015 by Carl Edward Rasmussen, Marc Deisenroth and
%                                 Andrew McHutchon, Rowan McAllister 2016-05-25

% 0. Incorporate dynmodel training into the controller
ctrl.set_dynmodel(dyn);                      % for CtrlBF. No effect for CtrlNF
ctrl.set_on(diag(exp(2*dyn.on)));
if ~exist('expl','var'); expl = []; end
if ~exist('cc_prev','var'); cc_prev.m = (H+1)*cost.MAX_COST; cc_prev.s = 1; end

% 1. Select best policy parameters to initalise optimisation from
losses = nan(j-1,1);
for i = 1:j-1 % loop over previous epsidoes' polices (current episode is 'j')
  losses(i) = loss(policies{i}, s, dyn, ctrl, cost, H, expl, cc_prev, N-j+1);
end
if ~isempty(losses)
  [~, i] = min(losses);
  ctrl.set_policy_p(policies{i});
end

% 2. Optimize policy
[p, fX3, ~, pp] = minimize(ctrl.policy.p, @loss, ctrl.policy.opt, s, dyn, ...
  ctrl, cost, H, expl, cc_prev, N-j+1); %#ok<IJCL>
policies{j} = p;
ctrl.set_policy_p(p);
% ctrl.set_policy_opt(pp); Apparently this doesn't work
% 2b. Exploration reference point to improve over next time:
if ~isempty(expl); [~, ~, ~, cc_prev] = simulate(s, dyn, ctrl, cost, H, expl.ccs_cov); end

% 3. Draw policy training curve (on figure 2)
if ~ishandle(2); figure(2); else set(0,'CurrentFigure',2); end
hold on; plot(fX3); drawnow;

% 4. Draw predicted cost (on figure 3)
[pred(j).state,pred(j).action,pred(j).cost] = simulate(s, dyn, ctrl, cost, H, false);
if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end
clf(3); errorbar(0:H, [pred(j).cost.m], 2*sqrt([pred(j).cost.s]));
axis tight; grid; drawnow;
