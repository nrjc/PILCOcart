function [cc, dcc] = uncertaintyReductionGivenMeanFantasyDataset(S, A, ...
  cc_pre, dcc_pre, s, dyn, ctrl, cost, H, expl)
% This function estimates the reduction in uncertainty of the cumulative-cost
% given a fantasy dataset made up of the predictive-mean states from simulate.
% The purpose of this function is to provide a more accurate estimate of the
% reduction in loss-uncertainty than simply using 'cc_total' whose total
% uncertainty is composed of both system ignorance *and* inherent system
% stochasticity (i.e. process noise and observation noise). This function
% corresponds to Solution A in the  estimateUncertaintyReduction.pdf document.
%
% [cc, dcc] = uncertaintyReductionGivenMeanFantasyDataset(S, A, cc_pre, ...
%   dcc_pre, s, dyn, ctrl, cost, H)
%
% INPUTS:
% S         H+1   state struct array containing Gaussian state distributions
%   m       Dx1   mean vector
%   s       DxD   variance matrix
% A         H     action struct array containing Gaussian action distributions
%   m       Ux1   mean vector
%   s       UxU   variance matrix
% cc_pre          cumulative (discounted) cost structure
%   m             mean scalar
%   s             variance scalar
% dcc_pre         derivative structure of cc_pre
%   m       1xP   derivative cc_pre-mean wrt policy parameters
%   s       1xP   derivative cc_pre-variance wrt policy parameters
% s         .     state structure
% dyn       .     dynamics model object
% ctrl      .     controller object
% cost      .     cost function object
% H               length of prediction horizon
% expl            exploration struct
%   ccs_cov bool  flag, do we want to compute accurate-yet-expensive cross
%                 covariance terms in cumulative cost variance (cc.s)?
%
% OUTPUTS:
% cc              cumulative (discounted) cost structure
%   m             mean scalar
%   s             variance scalar
% dcc             derivative structure of cc
%   m       1xP   derivative cc-mean wrt policy parameters
%   s       1xP   derivative cc-variance wrt policy parameters
%
% See also <a href="estimateUncertaintyReduction.pdf">estimateUncertaintyReduction.pdf</a>.
% Copyright (C) 2016 Carl Edward Rasmussen and Rowan McAllister 2016-05-10

D = ctrl.D; E = dyn.E; na = numel(dyn.angi); nf = numel(S)-1;

% 0. Trim filter states from fantasy data based on the (fixed) current policy:
for i=1:numel(S)
  S(i).m = S(i).m(1:D);
  S(i).s = S(i).s(1:D,1:D);
end

% 1. Copy dynmodel, and append fantasy data
dync = dyn.deepcopy();
fan = [[S(1:end-1).m]', [A.m]', nan(nf,2*na)]; % FANtasy inputs, state + action
for i = 1:na                             % augment with trigonometric functions
  fan(:,end-2*na+2*i-1) = sin(fan(:,dyn.angi(i)));
  fan(:,end-2*na+2*i)   = cos(fan(:,dyn.angi(i)));
end
if numel(dyn.induce) > 0
  in = dyn.induce; pE = size(in,3);
  dync.induce = [in; repmat(fan,1,1,pE)];
else
  in = dyn.inputs; pE = size(in,3);
  dync.inputs = [in; repmat(fan,1,1,pE)];
end
target = [S(2:end).m]'; target = target(:,end-E+1:end);           % set targets
dync.target = [dyn.target; target];

y = dync.target - ...             % y is targets less linear contribution
  bsxfun(@plus,sum(bsxfun(@times,permute(dync.inputs,[1,3,2]),...
  permute([dync.hyp.m],[3,2,1])),3), [dync.hyp.b]);

% 2. Compute "pre" variables in an efficient way
n = size(dync.target,1); dync.W = nan(n,n,E); dync.beta = nan(n,E);
for i = 1:E
  h = dyn.hyp(i);
  W = dyn.W(:,:,i);
  x = bsxfun(@times, in(:,:,min(i,pE)), exp(-h.l'));
  xf = bsxfun(@times, fan, exp(-h.l'));
  kf  = exp(2*h.s-maha(x,xf)/2);
  kff = exp(2*h.s-maha(xf,xf)/2) + exp(2*h.n)*eye(nf);
  
  % Block matrix inversion (see page 201 eqs A.11-12 of GPML book)
  im = kff - kf'*W*kf;
  pt = W + W*kf/im*kf'*W;
  qt = -W*kf/im;
  st = inv(im);
  dync.W(:,:,i) = [pt , qt ; qt' , st];
  
  dync.beta(:,i) = dync.W(:,:,i) * y(:,i);
end

% 3. Call simulate with modified dynamics model to estimate reduced uncertainty
ctrldyn = ctrl.dyn;
ctrl.set_dynmodel(dync);
assert(~any(isnan(unwrap(dync.beta))));
assert(~any(isnan(unwrap(dync.inputs))));
if nargout < 2 % no derivatives
  [~, ~, ~, ccf] = simulate(s, dync, ctrl, cost, H, expl.ccs_cov);
else
  [~, ~, ~, ccf, dccf] = simulate(s, dync, ctrl, cost, H, expl.ccs_cov);
end
assert(~any(isnan(unwrap(dync.beta))));
assert(~any(isnan(unwrap(dync.inputs))));
ctrl.set_dynmodel(ctrldyn); % reset controller

% 4. Compute uncertainty reduction, given pre and post uncertainty
cc.m = ccf.m;      % use an anticipated change of expectation, not expected pre
cc.s = cc_pre.s - ccf.s;
if nargout > 1
  dcc.m = dccf.m;
  dcc.s = dcc_pre.s - dccf.s;
end
if cc.s < 0;
  % keyboard % TODO: decide how to handle cc.s < 0, perhaps a squashing function to keep it always positive.
  disp([mfilename,': cc.s < 0. Setting cc.s = 0.']);
  cc.s = 1e-4; % 0 can cause problems with checkgrad loss
  if nargout > 1; dcc.s = 0*dcc.s; end
end

assert(~any(isnan(unwrap(cc))));
if nargout > 1
  assert(~any(isnan(unwrap(dcc))));
end
