function ctrlBFT(dyn, ctrl, s, delta)

% Test the ctrlBF function. Check the three outputs using Monte Carlo, and the
% derivatives using finite differences.
%
% CTRLBFT(dyn, ctrl, s, delta):
%   dyn           dynamics model object
%   ctrl          controller object
%   s             state structure
%   delta         finite difference (default 1e-5)
%
% See also <a href="ctrlBF.pdf">ctrlBF.pdf</a>, CTRLBF.M.
% Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2015-09-24

clc
NSAMPLES = 3e3;
SEED = 7;
EPSILON = 1e-5;               % 'pass' threshold for low enough checkgrad error
rng(SEED);
addpath('../control'); addpath('../util'); addpath('../gp'); addpath('../base')
dbstop if error
EXPERIMENT = 'cartPoleMarkov'; %'cartDoublePendulum';  % easy, medium, hard
RESET_THE_FILTER = false;
% USE_APPROX_CTRLBF = true;

% Gradients to test:
douts ={'M', 'S', 'C', 'm', 's', 'v'};
dins ={'m', 's', 'v', 'p'};
%douts = {'t'}; dins = {'v'}; % aribtary test varible
% douts ={'S', 'v'};
douts ={'t'};
dins ={'v'};

% 1. SET CTRLBF INPUTS --------------------------------------------------------

% default inputs:
plant = create_test_object('plant', EXPERIMENT);
if ~exist('dyn','var'); dyn = create_test_object('dyn', plant); end
if ~exist('ctrl','var'); ctrl = create_test_object('CtrlBF', plant, dyn); end
assert(isa(ctrl,'CtrlBF'), 'ctrlBFT: controller input muct be class CtrlBF');
D = ctrl.D; E = ctrl.E; F = ctrl.F; assert(F==2*D); U = ctrl.U; onp = ctrl.onp;
if ~exist('s','var')
  s.m = randn(D,1)+2;
  s.m = [s.m ; s.m + 0.1*randn(D,1)];
  s.s = 0.5*symrandn(2*D);
  s.v = 2.0*onp;
end
if RESET_THE_FILTER
  s = ctrl.reset_filter(s);
else
  s.reset = false;
end
assert(ctrl.F == length(s.m));
assert(all(eig(s.s) >= -1e-12));
if ~exist('delta','var'); delta = 1e-5; end     % checkgrad's finite difference

% 2. TEST CTRLBF OUTPUT-DERIVATIVES -------------------------------------------

args = {s};
ntests = numel(dins)*numel(douts);
test_names = cell(ntests,1); cg = cell(ntests,1); i = 0;
for din = dins;
  for dout = douts;
    i = i+1;
    test_names{i} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{i} = cg_wrap(args{:}, ctrl.policy.p, ctrl, nan, delta, dout{:}, din{:});
  end
end
print_derivative_test_results(test_names, cg, EPSILON)

if any(strcmp(dins,'v')) && max(s.v(:)) > 100;
  disp('ctrlBFT: s.v is large, so finite-diff dXdv gradients is inaccurate');
end

if any(strcmp(douts,'t'));
  disp(['ctrlBFT: CtrlBF is in gradient Test mode only (has a s.t field), ' ...
    'so returning now.']); return
end

% 3. TEST CTRLBF OUTPUTS ------------------------------------------------------

% 3.1. Compute Analytic Outputs
assert(all(eig(s.s)>=-1e-8));
[Ma_, Sa_, isCa_, sa_,~,~,~,~,~,~,~,~] = ctrl.fcn(args{:});
[Ma , Sa , isCa,  sa] = ctrl.fcn(args{:});
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({Ma_,Sa_,isCa_,sa_}, {Ma,Sa,isCa,sa}) < 1e-5, ...
  'ctrlBFT: requesting derivatives shold not alter non-derivative outputs.');
sva = sa.v;
Ca = s.s*isCa;

% 3.2. Compute Numeric Outputs
Z = zeros(D);
x = mvnrnd(s.m,s.s,NSAMPLES);               % noise-free samples
y = x + mvnrnd(0*s.m,[onp,Z;Z,Z],NSAMPLES); % noisy samples
Mi = nan(NSAMPLES,U+D); Si = nan(NSAMPLES,U+D,U+D); vi = nan(NSAMPLES,D,D);
% si.s = 0*s.s; % propagate mode?
si.v = s.v;
si.reset = s.reset;
for i=1:NSAMPLES
  print_loop_progress(i,NSAMPLES,'Testing outputs with MC');
  si.m = y(i,:)';
  [Mi(i,:), Si(i,:,:), ~, si_next] = ctrl.fcn(si);
  vi(i,:,:) = si_next.v;
end
Mn = mean(Mi,1)';
Sn = cov(Mi); % + squeeze(mean(uSi,1));
Cn = nan(F,U+D);
svn = mean(vi,1);
for f=1:F
  for ud=1:U+D; c=cov(x(:,f)',Mi(:,ud)); Cn(f,ud)=c(1,2); end
end
fprintf('\nctrlBFT: uSi has a maximium value of %f\n', max(abs(Si(:))));

% 3.3 Dislpay numeric vs. analytic
str = @(x) (num2str(unwrap(x)'));
u = 1:U; z = U+1:U+D;
fprintf('\nCTRLBF MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================================================\n');
fprintf('M (u)    numeric  : %s\n', str(Mn(u)));
fprintf('M (u)    analytic : %s\n', str(Ma(u)));
fprintf('--------------------\n');
fprintf('M (z)    numeric  : %s\n', str(Mn(z)));
fprintf('M (z)    analytic : %s\n', str(Ma(z)));
fprintf('--------------------\n');
fprintf('S (uu)   numeric  : %s\n', str(Sn(u,u)));
fprintf('S (uu)   analytic : %s\n', str(Sa(u,u)));
fprintf('--------------------\n');
fprintf('S (uz)   numeric  : %s\n', str(Sn(u,z)));
fprintf('S (uz)   analytic : %s\n', str(Sa(u,z)));
fprintf('--------------------\n');
fprintf('S (zu)   numeric  : %s\n', str(Sn(z,u)));
fprintf('S (zu)   analytic : %s\n', str(Sa(z,u)));
fprintf('--------------------\n');
fprintf('S (zz)   numeric  : %s\n', str(Sn(z,z)));
fprintf('S (zz)   analytic : %s\n', str(Sa(z,z)));
fprintf('--------------------\n');
fprintf('C (u)    numeric  : %s\n', str(Cn(:,u)));
fprintf('C (u)    analytic : %s\n', str(Ca(:,u)));
fprintf('--------------------\n');
fprintf('C (z)    numeric  : %s\n', str(Cn(:,z)));
fprintf('C (z)    analytic : %s\n', str(Ca(:,z)));
fprintf('--------------------\n');
fprintf('s.v      numeric  : %s\n', str(svn));
fprintf('s.v      analytic : %s\n', str(sva));
fprintf('--------------------\n');
fprintf('==================================================\n');
mdiff = max_diff({Mn,Sn,Cn,svn}, {Ma,Sa,Ca,sva});
fprintf('Maximum difference = %4.2e\n', mdiff);

%fprintf('\nctrlBFT: controller in approx mode [%u] only \n', ctrl.approxZC);

% 4. FUNCTIONS ----------------------------------------------------------------

% Generate a random square symmetric matrix
function r = symrandn(D)
r = randn(D); r = r'*r;

% args = {s_noisy};
% cg_wrap(args{:}, orig_policyp, ctrl, nan, DELTA, dout{:}, din{:})));

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
function cg = cg_wrap(varargin)
[orig_policyp, ~, ~, delta, dout, din] = deal(varargin{end-5:end});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
if din == 'p', x = orig_policyp; else x = varargin{1}.(din); end
[d,dy,dh] = checkgrad(@cg_f,x,delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
% varargin: fcn inputs: ctrl.fnc(s)
function [f, df] = cg_f(x,varargin)
[orig_policyp,ctrl,~,~,dout,din] = deal(varargin{end-5:end});
in = varargin(1:end-6); is = ctrl.is;
if any(strcmp(din,{'s','v'})); x = (x+x')/2; end
% in{1}.computeZC = strcmp(dout,'zc');
if strcmp(din,'p'), ctrl.set_policy_p(x); else in{1}.(din) = x; end
if nargout == 1
  [M, S, C, s] = ctrl.fcn(in{:});
else
  [M,S,C,s,dMds,dSds,dCds,dsds,dMdp,dSdp,dCdp,dsdp] = ctrl.fcn(in{:});
end
switch dout
  % t  :  arbitary variable for ctrlBFT.m gradient-testing
  % dt :  derivative of test variable t
  case 't'; f = s.t;
    if nargout == 2; df = s.dt; end
  case 'M'; f = M;
    if nargout == 2
      if din == 'p', df = dMdp; else df = dMds(:,is.(din)); end
    end
  case 'S'; f = S;
    if nargout == 2
      if din == 'p', df = dSdp; else df = dSds(:,is.(din)); end
    end
  case 'C'; f = C;
    if nargout == 2
      if din == 'p', df = dCdp; else df = dCds(:,is.(din)); end
    end
  otherwise; f = s.(dout);
    if nargout == 2
      if din == 'p', df = dsdp(is.(dout),:);
      else df = dsds(is.(dout),is.(din)); end
    end
end
ctrl.set_policy_p(orig_policyp);  % reset