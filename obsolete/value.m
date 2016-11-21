function [f, df] = value(p, s, dyn, ctrl, cost, H)

% [f, df] = value(p, s, dyn, ctrl, cost, H)
%
% p         .    struct of policy parameters chosen by minimize
% s         .    state structure
%   m            mean vector
%   s            covariance matrix
%   ?            possibly other fields representing additional information
% dyn       .    dynamics model object
% ctrl      .    controller object
%   is      .    struct indexing vectorized state distributions variables
%     m          indices of mean parameters
%     s          indices of variance parameters
%   np      s    number of parameters in the policy
%   ns      s    number of state distribution parameters (means and variances)
%   policy  .    policy structure
%     p     .    policy parameters structure
% cost      .    cost function structure
%   fcn     @    function implementing cost
%   gamma   s    discount factor
% H         s    length of prediction horizon
% f         .    state action loss struct -- only if isempty(p)
%   state   H+1  state struct array containing Gaussian state distributions
%     m          mean vector
%     s          covariance matrix
%   action  H    action struct array containing Gaussian action distributions
%     m          mean vector
%     s          covariance matrix
%   cost    H+1  cost struct array containing Gaussian cost distributions
%     m     s    mean scalar  
%     s     s    variance scalar
% f         s    expected cumulative (discounted) cost -- only if ~isempty(p)
% df        .    derivative struct of f wrt policy parameters, same fields as p
%
% Copyright (C) 2008-2015 Carl Edward Rasmussen, 2015-03-20

if isempty(p)      % if policy is empty, then compute state, actions and losses
  f.state(1) = s; [f.cost(1).m, f.cost(1).s] = cost.fcn(cost, s);      % init f
  for t = 1:H                                           % iterate up to horizon
    [f.state(t+1), f.action(t)] = propagate(f.state(t), dyn, ctrl); % propagate
    [f.cost(t+1).m, f.cost(t+1).s] = cost.fcn(cost, f.state(t+1));  % calc cost
  end
else                   % otherwise compute losses and posibly their derivatives
  f = 0; ctrl.set_policy_p(p);    % zero loss and set policy parameters in ctrl
  if nargout == 1                                  % losses only, no derivaties
    for t = 1:H                                 % for all time steps in horizon
      s = propagate(s, dyn, ctrl);       % propagate state distribution forward
      f = f + cost.gamma^t*cost.fcn(cost, s);      % accumulate discounted cost
    end
  else                                       % otherwise losses and derivatives
    dp = zeros(ctrl.np,1); sdp = zeros(ctrl.ns,ctrl.np);     % init derivatives
    for t = 1:H                                         % iterate up to horizon
      [s, dsds, dsdp] = propagated(s, dyn, ctrl);           % propagate forward
      sdp = dsds*sdp + dsdp;                                       % chain rule
      [L, ~, dLds] = cost.fcn(cost, s);         % cost and derivative wrt state
      f = f + cost.gamma^t*L;                      % accumulate discounted cost
      dp = dp + cost.gamma^t*(unwrap(dLds)'*sdp([ctrl.is.m; ctrl.is.s(:)],:))';
    end
    df = rewrap(ctrl.policy.p, dp);        % rewrap derivative to original type
  end
end
