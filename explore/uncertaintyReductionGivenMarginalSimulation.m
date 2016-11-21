function [cc, dcc] = uncertaintyReductionGivenMarginalSimulation(S, A, ...
  cc_pre, dcc_pre, s, dyn, ctrl, cost, H, expl)
% This function estimates the reduction in uncertainty of the cumulative-cost
% given a the simulate's prediction of the sequence of state distributions 
% defines the marginal distribution of the state predictions accross all 
% possbile fantasy datasets we might see next.
% The purpose of this function is to provide a more accurate estimate of the
% reduction in loss-uncertainty than simply using 'cc_total' whose total
% uncertainty is composed of both system ignorance *and* inherent system
% stochasticity (i.e. process noise and observation noise). This function
% corresponds to Solution A in the  estimateUncertaintyReduction.pdf document.
%
% [cc, dcc] = uncertaintyReductionGivenMarginalSimulation(S, A, cc_pre, ...
%   dcc_pre, s, dyn, ctrl, cost, H)
%
% INPUTS:
% S         H+1   state struct array containing Gaussian state distributions
%   m       Dx1   mean vector
%   s       DxD   variance matrix
% A         H     action struct array containing Gaussian action distributions
%   m       Ux1   mean vector
%   s       UxU   variance matrix
% cc_pre          cumulative (discounted) cost structure before fantasy data
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
% See also and <a href="expected-variance-given-fantasy-data.pdf">expected-variance-given-fantasy-data.pdf</a>
% <a href="estimateUncertaintyReduction.pdf">estimateUncertaintyReduction.pdf</a>.
% Copyright (C) 2016 Carl Edward Rasmussen, Rowan McAllister and 
% Mark van der Wilk 2016-05-10

% Handle inputs ---------------------------------------------------------------

D = ctrl.D; E = dyn.E; na = numel(dyn.angi); nS = numel(S);
nf = nS-1; % number of fantasy data (could be < H if state diverged)
derivatives_requested = nargout > 1;

% 0. Trim filter states from fantasy data based on the (fixed) current policy:
for i = 1:nS
  S(i).m = S(i).m(1:D);
  S(i).s = S(i).s(1:D,1:D);
end

% 1. Copy dynmodel, and append fantasy data
dync = dyn.deepcopy();
fan = [[S(1:end-1).m]', [A.m]', nan(nf,2*na)]; % FANtasy inputs, state + action
assert(size(fan,1) == nf);
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
% targets will probably not be used, but let's set them to be safe:
% target = bsxfun(@plus,fan*[dync.hyp.m], [dync.hyp.b]);                  
% target = target(:,end-E+1:end);           % set targets               
% dync.target = [dyn.target; target];
dync.target = nan; % varify targets are not used in gp prediction.
%y = dync.target - ...             % y is targets less linear contribution
%  bsxfun(@plus,sum(bsxfun(@times,permute(dync.inputs,[1,3,2]),...
%  permute([dync.hyp.m],[3,2,1])),3), [dync.hyp.b]);
%assert(all(max(abs(y(end-nf+1:end,:))) < 1e-12)); % y(end-nfan+1:end,:) = 0;     
assert(~any(isnan(unwrap({dyn.beta,dyn.inputs,dyn.W})))); 
% 2. Compute "pre" variables in an efficient way
n = size(dync.inputs,1); dync.W = nan(n,n,E); dync.beta = nan(n,E);
for i = 1:E
  h = dyn.hyp(i);
  W = dyn.W(:,:,i);
  x = bsxfun(@times, in(:,:,min(i,pE)), exp(-h.l'));
  xf = bsxfun(@times, fan, exp(-h.l'));
  kf  = exp(2*h.s-maha(x,xf)/2);
  kff = exp(2*h.s-maha(xf,xf)/2) + exp(2*h.n)*eye(nf); % TODO: Mark update noise?
  
  % Block matrix inversion (see page 201 eqs A.11-12 of GPML book)
  im = kff - kf'*W*kf;
  pt = W + W*kf/im*kf'*W;
  qt = -W*kf/im;
  st = inv(im);
  dync.W(:,:,i) = [pt , qt ; qt' , st];
  % dync.beta(:,i) = dync.W(:,:,i) * y(:,i); % incorrect.
end
dync.beta = [dyn.beta; zeros(n-size(dyn.beta,1),size(dyn.beta,2))]; % TODO: is this correct?

% check nan bugs:
% assert(~any(isnan(unwrap({dync.beta,dync.inputs,dync.W})))); 
if any(isnan(unwrap({dync.beta,dync.inputs,dync.W})))
  cc.m = cc_pre.m; % TODO: check sum([L.m]) == cc_pre.m.
  cc.s = 1e-4; % variance reduction in Loss.
  if derivatives_requested; 
    dcc.m = dcc_pre.m;
    dcc.s = 0*dcc.s;
  end
  warning('[estimate_uncertainty_reduction2]: cc.s < 0. Setting cc.s = 0.');
end

% 3. Re-do each section of simulate with modified dynamics model to estimate reduced uncertainty
S2 = S; % alloc, and note S2(1) == S(1).
sdp  = cell(nS,1); sdp{1}  = zeros(ctrl.ns,ctrl.np); % original run with just usual data
sdp2 = cell(nS,1); sdp2{1} = zeros(ctrl.ns,ctrl.np); % new run with usual data PLUS fantasy
if ~derivatives_requested
  for i = 1:nS-1
    % S2(i+1) = propagate(S(i), dync, ctrl); % TODO: is this correct, or below?
    S2(i+1) = propagate(S2(i), dync, ctrl); % cumilative effect
    S2(i+1).m = S(i+1).m; % its location remains fixed?
    assert(all(eig(S2(i+1).s))>-1e-1);
  end
else
  for i = 1:nS-1
    % dyn prop:
    [~, ~, ~, dsds, dsdp] = propagated(S(i), dyn, ctrl);
    sdp{i+1} = dsds*sdp{i} + dsdp;
    % dync prop:
    % [S2(i+1), ~, ~, dsds2, dsdp2] = propagated(S(i), dync, ctrl); % TODO: is this correct, or below?
    [S2(i+1), ~, ~, dsds2, dsdp2] = propagated(S2(i), dync, ctrl);
    S2(i+1).m = S(i+1).m; % its location remains fixed?
    % dsds2(ctrl.is.m,:) = dsds(ctrl.is.m,:); % unsure if correct?
    sdp2{i+1} = dsds2*sdp2{i} + dsdp2;
    sdp2{i+1}(ctrl.is.m,:) = sdp{i+1}(ctrl.is.m,:);
    assert(all(eig(S2(i+1).s))>-1e-1);
  end
end

% TESTING:
% cc.m = 0;
% cc.s = 0;
% dcc.m = zeros(1,ctrl.np);
% dcc.s = zeros(1,ctrl.np);
% for i = 1:nS
%   cc.m=cc.m+sum(S2(i).m(:) + S(i).m(:));
%   cc.s=cc.s+sum(S2(i).s(:) + S(i).s(:));
%   if derivatives_requested
%     dcc.m=dcc.m+sum(sdp2{i}(ctrl.is.m,:) + sdp{i}(ctrl.is.m,:),1);
%     dcc.s=dcc.s+sum(sdp2{i}(ctrl.is.s,:) + sdp{i}(ctrl.is.s,:),1);
%   end
% end
% return

% Handle outputs --------------------------------------------------------------

% L = nan(H+1,1);
dLm_dp = nan(1,ctrl.np);
dLs_dp = nan(1,ctrl.np);
bad_eig_printed = false;
for t=1:nS
  % state-hierarchical:
  %stateh.m = S2(t).m; % TODO: S2 or S? Different means? (not anymore)
  stateh.m = S(t).m; % TODO: S2 or S? Them might have different means? If change here, then change gradient below.
  stateh.s = S(t).s - S2(t).s; % variance-mean == variance-reduction-amount
  stateh.v = S2(t).s; % new-(reduced)-variance
  
  bad_eig = any(eig(stateh.s)<-1e-1);
  if bad_eig % && t>1
    if ~bad_eig_printed
      disp([mfilename,': eig(stateh.s)<0 at t',num2str(t)]);
      bad_eig_printed = true;
    end
    stateh.s = zeros(D); %0*stateh.s; % cannot enlarge grater than prior variance.
  end
  
  if ~derivatives_requested
    L(t) = cost.fcnh(stateh); 
  else
    [L(t), dL(t)] = cost.fcnh(stateh);
    is = ctrl.is;
    % dstateh_dp = [sdp2{t}(is.m,:); flag_pos_reduction*(sdp{t}(is.s,:) - sdp2{t}(is.s,:)); sdp2{t}(is.s,:)];
    dstateh_dp = [sdp{t}(is.m,:); ~bad_eig*(sdp{t}(is.s,:) - sdp2{t}(is.s,:)); sdp2{t}(is.s,:)];
    dLm_dp(t,:) = dL(t).m * dstateh_dp;
    dLs_dp(t,:) = dL(t).s * dstateh_dp;
  end
  
%   if bad_eig
%     % fill in the rest if we broke away early:
%     for tt = length(L)+1:H+1
%       L(tt).m = cost.MAX_COST;
%       L(tt).s = 0;
%       dL(tt).m = zeros(1,ctrl.np);
%       dL(tt).s = zeros(1,ctrl.np);
%     end
%     break
%   end
end

% Outputs
cc.m = cc_pre.m; % TODO: check sum([L.m]) == cc_pre.m.
% assert(abs(sum([L.m]) - cc_pre.m) < 1e-5);
cc.s = sum([L.s]); % variance reduction in Loss.
% and we do nothing with L.v. 
% TODO: perhaps check if sum([L.v]) + sum([L.s]) == cc_pre.s?

% disp('Lm, Ls, Lv, .s:')
% sum([L.m])
% sum([L.s])
% sum([L.v])
% cc_pre.s

% Derivative Outputs
if derivatives_requested
  %dcc.m = sum(dLm_dp,1); % is this correct?
  dcc.m = dcc_pre.m;
  dcc.s = sum(dLs_dp,1);
end
if cc.s <= 0;
  % TODO: decide how to handle cc.s < 0, perhaps a squashing function to keep it always positive.
  warning('[estimate_uncertainty_reduction]: cc.s < 0. Setting cc.s = 0.');
  cc.s = 1e-6;
  if derivatives_requested; dcc.s = 0*dcc.s; end
end

% check nan bugs:
assert(~any(isnan(unwrap(cc))));
if derivatives_requested; assert(~any(isnan(unwrap(dcc)))); end

%assert(abs(cc_pre.m - cc.m) < 1e-7); % TODO does this need to be checked?


