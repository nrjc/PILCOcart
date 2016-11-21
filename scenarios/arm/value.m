function [f, df] = value(p, m0, S0, dynmodel, ctrl, plant, cost, H)

% Compute expected (discounted) cumulative cost for a given (set of k) initial
% state distributions
%
% p                policy parameters chosen by minimize
% ctrl             controller structure
% m0        F x k  matrix oof initial state means
% S0        F x F  covariance matrix for initial state
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
% Copyright (C) 2008-2014 Marc Deisenroth & Carl Edward Rasmussen, 2014-04-03

ctrl.policy.p = p; % overwrite local policy.p with new parameters from minimize
p = unwrap(ctrl.policy.p); dp = 0*p;
ctrl.state = ctrl.init;                       % initialize the controller state
m = m0; S = S0; L = ones(1,H); D = length(plant.dyno);

if nargout <= 1                                       % no derivatives required
  for t = 1:H                                   % for all time steps in horizon
    ctrl.policy.t = t;
    [ctrl.state, m, S] = plant.prop(m, S, plant, dynmodel, ctrl);  % next state
    L(t) = cost.gamma^t.*cost.fcn(cost, m(1:D), S(1:D,1:D));     % compute cost
  end
else                                               % otherwise, get derivatives
  dmOdp = zeros([size(m0,1), length(p)]);
  dSOdp = zeros([size(m0,1)*size(m0,1), length(p)]);
    
  for t = 1:H                                   % for all time steps in horizon
    ctrl.policy.t = t;
    [ctrl.state, m, S, dmdmO, dSdmO, dmdSO, dSdSO, dmdp, dSdp] = ...
                    plant.prop(m, S, plant, dynmodel, ctrl);   % get next state
    if any(isnan(S(:))) || any(~isreal(S(:))) || min(eig(S)) > 1e10; break; end
    dmdp = dmdmO*dmOdp + dmdSO*dSOdp + dmdp;
    dSdp = dSdmO*dmOdp + dSdSO*dSOdp + dSdp;
      
    [L(t), dLdm, dLdS] = cost.fcn(cost, m(1:D), S(1:D,1:D));             % cost
    L(t) = cost.gamma^t*L(t);                                        % discount
    dp = dp + cost.gamma^t*( dLdm(:)'*dmdp + dLdS(:)'*dSdp )';
    dmOdp = dmdp; dSOdp = dSdp;                                   % bookkeeping
   end
end
    
f = sum(L); df = rewrap(ctrl.policy.p, dp);                  
