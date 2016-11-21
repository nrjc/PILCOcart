function [s, a, c, cc, dcc] = simulate(s, dyn, ctrl, cost, H, cc_cov_req)

% [s, a, c, cc, dcc] = simulate(s, dyn, ctrl, cost, H, ccs_req)
%
% INPUTS:
% s          .    state structure
%   m        F*1  mean vector
%   s        F*F  covariance matrix
%   ?             possibly other fields representing additional information
% dyn        .    dynamics model object
% ctrl       .    controller object
%   is       .    struct indexing vectorized state distributions variables
%     m      F*1  indices of mean parameters
%     s      F*F  indices of variance parameters
%   np            number of parameters in the policy
%   ns            number of state distribution parameters (means and variances)
%   policy   .    policy structure
%     p      .    policy parameters structure
% cost       .    cost function object
%   cov      @    function returning cost-covariances between two states
%   fcn      @    function returning expected and variance of a states's cost
%   gamma         discount factor
% H               length of prediction horizon
% cc_cov_req bool flag: is it required to compute (accurate yet expensive) 
%                 cross-covariance terms within cc.s?
%
% OUTPUTS:
% s          H+1  state struct array containing Gaussian state distributions
%   m        F*1  mean vector
%   s        F*F  covariance matrix
% a          H    action struct array containing Gaussian action distributions
%   m        U*1  mean vector
%   s        U*U  covariance matrix
% c          H+1  cost struct array containing Gaussian cost distributions
%   m        1x1  mean scalar
%   s        1x1  variance scalar
%   c        Dx1  inverse-input-variance times input-output covariance
% cc              cumulative (discounted) cost structure
%   m             mean scalar
%   s             variance scalar
% dcc             derivative structure of cc
%   m        1xP  derivative cc-mean wrt policy parameters, same fields as p
%   s        1xP  derivative cc-variance wrt policy parameters,same fields as p
%
% Copyright (C) 2008-2015 Carl Edward Rasmussen, 2016-03-04

global currT; currT=1;
if ~isfield(s,'s'); s.s = zeros(length(s.m)); end
c = cost.fcn(s);                                                      % init c
cc.m = c.m; cc.s = c.s;                                               % init cc
q = s.s;                                                              % init q
gamma = cost.gamma; D = ctrl.D; F = length(s.m);
is_state_diverging = @(s) (max(abs(unwrap({s.m, sqrt(s.s)}))) > 1e3);
state_diverged = false;
if nargout < 5 % no derivatives
  for t = 1:H                                         % iterate up to horizon
    try
      [s(t+1), C, a(t)] = propagate(s(t), dyn, ctrl); % propagate state forward
      state_diverged = is_state_diverging(s(t+1));
    catch 
      state_diverged = true;
    end
    if state_diverged; break; end
    c(t+1) = cost.fcn(s(t+1));                        % calc cost distribution
    if nargout > 3 % then accumulate discounted cost
      cc.m = cc.m + gamma^t*c(t+1).m;
      cc.s = cc.s + gamma^(2*t)*c(t+1).s;
      % cross-covariance terms:
      if ~cc_cov_req; continue; end
      q = [q*C ; s(t+1).s];
      for j = 1:t
        J = (j-1)*F+(1:D); % j'th row-block of q (non-filter vars)
        cc.s = cc.s + 2*gamma^(j+t-1)*cost.cov(s(j), s(t+1), q(J,1:D));
      end
    end
  end
else % do derivatives
  is = ctrl.is;
  ic = unwrap([is.m(1:D),is.s(1:D,1:D)]);   % cost indices, depend on real vars
  sdp = cell(H+1); sdp{1} = zeros(ctrl.ns,ctrl.np);          % init derivatives
  qdp = sdp{1}(is.s,:);
  dp = zeros(2,ctrl.np);              % first row = mean, second row = variance
  for t = 1:H                                           % iterate up to horizon
    try
      [s(t+1), C, a(t), dsds, dsdp, dCds, dCdp] = propagated(s(t), dyn, ctrl);
      state_diverged = is_state_diverging(s(t+1));
    catch
      state_diverged = true;
    end
    if state_diverged; break; end
    [c(t+1), dcds] = cost.fcn(s(t+1));          % cost and derivative wrt state
    cc.m = cc.m + gamma^t*c(t+1).m;
    sdp{t+1} = dsds*sdp{t} + dsdp;                                 % chain rule
    dp = dp + cost.gamma^t*([dcds.m; dcds.s]*sdp{t+1}(ic,:)); % TODO verify gamma power
    cc.s = cc.s + gamma^(2*t)*c(t+1).s;
    % cross-covariance terms:
    if ~cc_cov_req; continue; end
    qC = q*C;
    dCdp = dCdp + dCds*sdp{t};
    qdp = catd(1, prodd(q,dCdp) + prodd([],qdp,C), sdp{t+1}(is.s,:), qC);
    q = cat(1, qC, s(t+1).s);
    for j = 1:t
      J = (j-1)*F+(1:D); % j'th row-block of q (non-filter vars)
      [cov, dcovdsj, dcovdst, dcovdq] = cost.cov(s(j), s(t+1), q(J,1:D));
      cc.s = cc.s + 2*gamma^(j+t-1)*cov;
      dcov = dcovdsj*sdp{j}(ic,:) + dcovdst*sdp{t+1}(ic,:) + ...
        dcovdq*qdp(sub2ind2(size(q,1),J,1:D),:);
      dp(2,:) = dp(2,:) + 2*gamma^(j+t-1)*dcov;
    end
  end
  dcc.m = dp(1,:);
  dcc.s = dp(2,:);
end

if state_diverged % then remaining state/actions are nonexistant, and max costs
  disp(['[simulate] state diverged @ t = ',num2str(t),'.'])
  if ~exist('a','var'); a = []; end
  for tt = t:H
    c(tt+1).m = cost.MAX_COST;
    c(tt+1).s = 0;
    cc.m = cc.m + gamma^tt * c(tt+1).m;
  end
end
