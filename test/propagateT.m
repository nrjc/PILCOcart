function propagateT(deriv, dyn, ctrl, s, delta)

% Test derivatives of the propagate functions,
%
% PROPAGATET(deriv, dyn, ctrl, s, delta):
%   deriv       cell-matrix of derivatives to test, e.g. {{'m','s'}{'p'}}
%   dyn         dynamics model
%   ctrl        controller object
%   s           state structure
%   delta       finite difference (default 1e-4)
%
% Copyright (C) 2015 Carl Edward Rasmussen and Rowan McAllister 2015-07-23

clc
NSAMPLES = 1e4;
SEED = 11;
ctrl_class = 'CtrlNF';                    % default ctrl class if no ctrl input
HORIZON = 1;                                   % (integer) must be 1 or greater
EXPERIMENT = 'unicycle';
EPSILON = 1e-5;               % 'pass' threshold for low enough checkgrad error

if nargin >= 3 && ~isempty(ctrl); ctrl_class = class(ctrl); end
rng(SEED);

% 1. INPUTS -------------------------------------------------------------------

% default inputs:
plant = create_test_object('plant', EXPERIMENT);
if ~exist('dyn','var'); dyn = create_test_object('dyn', plant); end
if ~exist('ctrl','var'); ctrl = create_test_object(ctrl_class, plant, dyn); end
if ~exist('s','var'); s = create_test_object('state', plant, ctrl); end
if ~exist('delta','var'); delta = 1e-4; end

if nargin < 1 || isempty(deriv)
  douts ={'m', 's', 'C'};
  dins ={'m', 's', 'p'};
  if strcmp(ctrl_class, 'CtrlBF')
    douts{end+1} = 'v';
    dins{end+1} = 'v';
  end
else
  douts = deriv{1}; dins = deriv{2};
end

% 2. DERIVATIVE TEST  ---------------------------------------------------------

args = {s, dyn, ctrl};
ntests = numel(dins)*numel(douts);
test = cell(ntests,1); cg = cell(ntests,1); i = 0;
for din = dins;
  for dout = douts;
    i = i+1;
    test{i} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{i} = cg_wrap(args{:}, ctrl.policy.p, HORIZON, delta, dout{:}, din{:});
  end
end
print_derivative_test_results(test, cg, EPSILON)

% 3. OUTPUT TEST  -------------------------------------------------------------
% TODO test a
% 3.1. Compute Analytic Outputs
[s1,isC,a] = propagate(args{:});
[s2,isC2,a2,~,~,~,~] = propagated(args{:});
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({s1,isC,a},{s2,isC2,a2}) < 1e-5, ...
  'propagateT: requesting derivatives shold not alter non-derivative outputs.');
assert(isreal(unwrap({s1,isC,a})), 'propagate outputs are imaginary!');
Ca = s.s*isC;

% 3.2. Compute Numeric Outputs
E = ctrl.E; F = ctrl.F; U = ctrl.U;
x = mvnrnd(s.m,s.s,NSAMPLES);      % noise-free samples
% y = x + mvnrnd(0*s.m,dyn.on,NSAMPLES); % noisy samples
si_next_m = nan(NSAMPLES,F);
si_next_s = nan(NSAMPLES,F,F);
Ci = nan(NSAMPLES,F,F);
ai_m = nan(NSAMPLES,U);
ai_s = nan(NSAMPLES,U,U);
si = s; si.s = 0*s.s; si.computeZC = false;
for i=1:NSAMPLES
  print_loop_progress(i,NSAMPLES,'Testing U outputs with MC');
  si.m = x(i,:)';
  [si_next, Ci(i,:,:), ai] = propagate(si, dyn, ctrl);
  si_next_m(i,:) = si_next.m;
  si_next_s(i,:,:) = si_next.s;
  ai_m(i,:) = ai.m;
  ai_s(i,:,:) = ai.s;
end
smn = mean(si_next_m);
ssn = cov(si_next_m) + squeeze(mean(si_next_s,1));
an.m = mean(ai_m);
an.s = cov(ai_m) + squeeze(mean(ai_s,1));
Cn = nan(F);
for f1=1:F, for f2=1:F, c=cov(x(:,f1)',si_next_m(:,f2)); Cn(f1,f2)=c(1,2); end; end

% 3.3. Dislpay numeric vs. analytic
str = @(x) (num2str(unwrap(x)'));
fprintf('\nCTRLNF.M MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================\n');
fprintf('s.m     numeric  : %s\n', str(smn));
fprintf('s.m     analytic : %s\n', str(s1.m));
fprintf('------------------\n');
fprintf('s.s     numeric  : %s\n', str(ssn));
fprintf('s.s     analytic : %s\n', str(s1.s));
fprintf('------------------\n');
fprintf('C       numeric  : %s\n', str(Cn));
fprintf('C       analytic : %s\n', str(Ca));
fprintf('------------------\n');
fprintf('a.m     numeric  : %s\n', str(an.m));
fprintf('a.m     analytic : %s\n', str(a.m));
fprintf('------------------\n');
fprintf('a.s     numeric  : %s\n', str(an.s));
fprintf('a.s     analytic : %s\n', str(a.s));
fprintf('==================\n');
mdiff = max_diff({smn,ssn,Cn,an}, {s1.m,s1.s,Ca,a});
fprintf('Maximum difference = %4.2e\n', mdiff);

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad,e.g:
% cg_wrap(s, dynmodel, ctrl, orig_policyp, horizon, delta, dout{:}, din{:})));
function cg = cg_wrap(varargin)
[orig_policyp, ~, delta, dout, din] = deal(varargin{end-4:end});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
if din == 'p', x = orig_policyp; else x = varargin{1}.(din); end
[d,dy,dh] = checkgrad(@cg_f,x,delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
% varargin: fcn inputs: propagated(s, dynmodel, ctrl)
function [f, df] = cg_f(x,varargin)
[ctrl,orig_policyp,horizon,~,dout,din] = deal(varargin{end-5:end});
in = varargin(1:end-5); is = ctrl.is;
if any(strcmp(din,{'s','v'})); x = (x+x')/2; end
if strcmp(din,'p'), ctrl.set_policy_p(x); else in{1}.(din) = x; end
if nargout == 1
  [s, C] = propagate(in{:});
  for time = 2:horizon
    in{1} = s;
    [s, C2] = propagate(in{:});
  end
else
  [s, C, ~, dsds, dsdp, dCds, dCdp] = propagated(in{:});
  for time = 2:horizon
    in{1} = s;
    [s, C2, ~, dsds2, dsdp2, dCds2, dCdp2] = propagated(in{:});
    dsds = dsds2 * dsds;
    dsdp = dsds2 * dsdp + dsdp2;
  end
  if dout == 'C'
    if din == 'p'; df = dCdp;
    else df = dCds(:,is.(din)); end
  else
    if din == 'p'; df = dsdp(is.(dout),:);
    else df = dsds(is.(dout),is.(din)); end
  end
end
if dout == 'C'; f = C;
else f = s.(dout); end
ctrl.set_policy_p(orig_policyp);  % reset