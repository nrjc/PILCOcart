function simulateT(s, dyn, ctrl, cost, H, cc_cov_req, delta)
%% Brief Description and Interface
% Summary: Test derivatives of the simulate function
%
% simulateT(s, dyn, ctrl, cost, H, ccs_req, delta)
%
% INPUTS:
%
%   s          state structure
%   dyn        GP dynamics model object
%   ctrl       controller object
%     p        policy parameters (can be a structure)
%   cost       cost structure
%   H          prediction horizon. Default: 4
%   cc_cov_req is it required to compute the cross term compnents of cc.s?
%   delta      finite difference parameter. Default: 1e-4
%
% OUTPUTS:
%
%   dd         relative error of analytical vs. finite difference gradient
%   dy         analytical gradient
%   dh         finite difference gradient
%
% Copyright (C) 2008-2015 by Marc Deisenroth, Andrew McHutchon, Joe Hall,
% Carl Edward Rasmussen, Rowan McAllister 2015-05-15

clc
EPSILON = 1e-5;               % 'pass' threshold for low enough checkgrad error
NSAMPLES = 1e2;
%rng(2);

% 1. INPUTS -------------------------------------------------------------------
plant = create_test_object('plant', 'unicycle');
if ~exist('dyn','var'); dyn = create_test_object('dyn', plant); end
if ~exist('ctrl','var'); ctrl = create_test_object('CtrlBF', plant, dyn); end
if ~exist('s','var'); s = create_test_object('state', plant, ctrl); end
if ~exist('cost','var'); cost = create_test_object('cost', plant); end
if ~exist('H','var'); H = 4; end
if ~exist('ccs_req','var'); cc_cov_req = 1; end
if ~exist('delta','var'); delta = 1e-4; end

% 2. DERIVATIVE TEST  ---------------------------------------------------------
tests = {'m','s'};
test_names = cell(numel(tests),1); cg = cell(numel(tests),1);
pprev = ctrl.policy.p; i = 0;
for k_ = tests
  k = k_{:};
  i = i+1;
  test_names{i} = strcat('d(cc.',k,')/dp: ');
  [d,dy,dh]= checkgrad(@wrap_simulate,pprev,delta,s,dyn,ctrl,cost,H,cc_cov_req,k);
  cg{i} = {d,dy,dh};
end
ctrl.set_policy_p(pprev);
print_derivative_test_results(test_names, cg, EPSILON);

% 3. OUTPUT TEST  -------------------------------------------------------------

% 3.1. Compute Analytic Outputs
[s1, a1, c1, cc1]    = simulate(s, dyn, ctrl, cost, H, cc_cov_req);
[s2, a2, c2, cc2, ~] = simulate(s, dyn, ctrl, cost, H, cc_cov_req);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({s1,a1,c1,cc1},{s2,a2,c2,cc2}) < 1e-9, ...
  'simulateT: requesting derivatives shold not alter non-derivative outputs.');
assert(max_diff(sum([c1.m]),cc1.m) < 1e-12, ...
  'simulateT: summation of costs error');

% 3.2. Compute Numeric Outputs
x = mvnrnd(s.m,s.s,NSAMPLES);      % noise-free samples
% y = x + mvnrnd(0*s.m,dyn.on,NSAMPLES); % noisy samples
ccim = nan(NSAMPLES,1);
ccis = nan(NSAMPLES,1);
si = s; si.s = 0*s.s;
for i=1:NSAMPLES
  print_loop_progress(i, NSAMPLES, 'Testing outputs with MC');
  si.m = x(i,:)';
  [~, ~, ci]    = simulate(si, dyn, ctrl, cost, H);
  ccim(i,:) = sum([ci.m]);
  ccis(i,:,:) = sum([ci.s]);
end
ccn.m = mean(ccim);
ccn.s = cov(ccim) + mean(ccis);

% 3.3. Dislpay numeric vs. analytic
str = @(x) (num2str(unwrap(x)'));
fprintf('\nCTRLNF.M MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================\n');
fprintf('cc.m     numeric  : %s\n', str(ccn.m));
fprintf('cc.m     analytic : %s\n', str(cc1.m));
fprintf('------------------\n');
fprintf('cc.s     numeric  : %s\n', str(ccn.s));
fprintf('cc.s     analytic : %s\n', str(cc1.s));
fprintf('==================\n');
fprintf('Maximum difference = %4.2e\n', max_diff(ccn, cc1));

% 4. FUNCTIONS ----------------------------------------------------------------

function [f, df] = wrap_simulate(p, s, dyn, ctrl, cost, H, cc_cov_req, k)
ctrl.set_policy_p(p);
if nargout < 2
  [~, ~, ~, cc] = simulate(s, dyn, ctrl, cost, H, cc_cov_req);
else
  [~, ~, ~, cc, dcc] = simulate(s, dyn, ctrl, cost, H, cc_cov_req);
end
f = cc.(k);
if nargout == 2
  df = unwrap(dcc.(k))';
end
