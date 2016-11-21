function [f, df] = value(p, m, S, dynmodel, ctrl, plant, cost, H)

% Compute expected (discounted) cumulative cost for a given initial state
% distributions over a horizon H (and optionally derivatives).
%
% p                policy parameters chosen by minimize
% ctrl             controller structure
% m         F x 1  vector of initial state means
% S         F x F  covariance matrix for initial state
% dynmodel         dynamics model structure
% plant            plant structure
% cost             cost function structure
%   fcn     @      function implementing cost
%   target  D x 1  vector of desired target values
%   gamma          discount factor
% H                length of prediction horizon
%
% f                expected cumulative (discounted) cost
% df               derivative struct of f wrt policy
%
% Copyright (C) 2008-2014 Marc Deisenroth & Carl Edward Rasmussen, 2014-04-25

ctrl.policy.p = p; % overwrite local policy.p with new parameters from minimize
p = unwrap(ctrl.policy.p); dp = 0*p; L = ones(1,H);
ctrl.state = ctrl.init;                       % initialize the controller state

if nargout < 2                                        % no derivatives required
  for t = 1:H                                   % for all time steps in horizon
    [ctrl.state, m, S] = plant.prop(m, S, plant, dynmodel, ctrl);  % next state
    L(t) = cost.gamma^t.*cost.fcn(cost, m, S);                   % compute cost
  end
else                                               % otherwise, get derivatives
  mdp = zeros([size(m,1), length(p)]); Sdp = zeros([numel(S), length(p)]);
    
  for t = 1:H                                   % for all time steps in horizon      
    [ctrl.state, m, S, dmdm, dSdm, dmdS, dSdS, dmdp, dSdp] = ...
                      plant.prop(m, S, plant, dynmodel, ctrl); % get next state
    if any(isnan(S(:))) || any(~isreal(S(:))) || min(eig(S)) > 1e10; break; end
    dmdp = dmdm*mdp + dmdS*Sdp + dmdp; dSdp = dSdm*mdp + dSdS*Sdp + dSdp;
      
    [L(t), dLdm, dLdS] = cost.fcn(cost, m, S);                           % cost
    L(t) = cost.gamma^t*L(t);                                        % discount
    dp = dp + cost.gamma^t*( dLdm(:)'*dmdp + dLdS(:)'*dSdp )';    % accum deriv
    mdp = dmdp; Sdp = dSdp;                                       % bookkeeping
  end
end
    
f = sum(L); df = rewrap(ctrl.policy.p, dp);                  
