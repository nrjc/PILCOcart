function [f, df] = lossZeroWeightDesc(p, s, dyn, ctrl, cost, H, expl, cc_prev, n)
% LOSS is the objective function we minimise to find a locally optimal policy
% parameterisation.
%
% If the heuristic function 'expl' is not inputted (or empty) then the expected
% cumulative cost is returned. If 'expl' is present, then the loss is a function
% of the cumulative cost mean and variance of the current policy
% parameterisation 'cc', and the policy parameterisation of the previous rollout
% 'cc_prev', and the number of rollout trials remaining 'n'. Derivatives of
% these quantities are computed if requested.
%
% Example call:
% [f, df] = loss(p, s, dyn, ctrl, cost, H, exp, cc_prev, n)
%
% INPUTS:
% p             policy parameter structure to evaluate
% s             initial state structure
% dyn           dynamics model object
% ctrl          controller object
%   policy      policy struct
%     p         policy parameters struct
% cost          cost object
% H             time-steps horizon
% expl          exploration struct
%   fcn         exploration heuristic function (Bayesian optimisation method)
%   method      method to compute the reduction in cumulative-cost variance cc.s
%   ccs_cov     bool flag, do we want to compute accurate-yet-expensive cross
%               covariance terms in cumulative cost variance (cc.s)?
% cc_prev       cumulative (discounted) cost struct from previous rollout
% n             number of trials remaining (inc. current point in time, so n>0)
%
% OUTPUTS:
% f        1x1  loss
% df       1xP  loss derivative wrt policy parameters
%
% Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2016-05-06

exploring = exist('expl','var') && ~isempty(expl);
if ~exist('cc_prev','var'); cc_prev.m = (H+1)*cost.MAX_COST; cc_prev.s = 1; end
if ~exist('n','var'); n = 1; end
% accurate computations for exploration unless user chose otherwise:
if exploring && ~isfield(expl,'ccs_cov'); expl.ccs_cov = true; end
if exploring; compute_ccs_cov = expl.ccs_cov; else compute_ccs_cov = false; end
% old_p = ctrl.policy.p; % record fixed policy
if ~isempty(p); ctrl.set_policy_p(p); end % evaluate controller at policy p

if nargout < 2 % no derivatives required
  [S, A, ~, cc] = simulate(s, dyn, ctrl, cost, H, compute_ccs_cov);
  if exploring
    cc_reduce = feval(expl.method, S, A, cc, [], s, dyn, ctrl, cost, H, expl);
    f = expl.fcn(expl, cc_reduce, cc_prev, n);
  else
    f = cc.m; % if not exploring then optimise w.r.t. cumulative-cost mean only
  end
else % derivatives required
  [S, A, ~, cc, dcc] = simulate(s, dyn, ctrl, cost, H, compute_ccs_cov);
  if exploring
    [cc_reduce, dcc_reduce] = feval(expl.method,S,A,cc,dcc,s,dyn,ctrl,cost,H,expl);
    [f, df] = expl.fcn(expl, cc_reduce, cc_prev, n);
    df = df.m * dcc_reduce.m + df.s * dcc_reduce.s;
    df(length(dcc.m)-length(ctrl.policy.p.w):end)=0; %Zeroing end of df, or the weight derivatives
  else
    f = cc.m; % if not exploring then optimise w.r.t. cumulative-cost mean only
    dcc.m(length(dcc.m)-length(ctrl.policy.p.w):end)=0; %Zeroing the end of the dcc.m,or the weight derivatives 
    df = dcc.m;
  end
end

% For checkgrad - under exploration - to generate samples from a fixed policy:
% if exploring && nargout < 2
%   ctrl.set_policy_p(old_p); % controller unchanged during finite differences
% end
