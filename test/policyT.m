function policyT(policy, m, s)

% Test the ctrlBF function. Check the three outputs using Monte Carlo, and the
% derivatives using finite differences.
%
% policyT(policy, m, s):
%   policy       policy function
%   m            (optional) input mean vector
%   s            (optional) input covariance matrix
%
% Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2015-06-06

clc
N = 1e4;                                              % monte Carlo sample size
DELTA = 1e-4;                                    % for finite difference approx
EPSILON = 1e-5;               % 'pass' threshold for low enough checkgrad error
TEST_DERIVATIVES = true;
SEED = 1;
rng(SEED);

if ~exist('policy', 'var')
 plant = create_test_object('plant','cartPoleMarkov');
 ctrl = create_test_object('CtrlNF', plant);
 policy = ctrl.policy;
end
if isfield(policy.p, 'inputs')
  D = size(policy.p.inputs,2);
elseif isfield(policy.p, 'w') 
  D = size(policy.p.w,2);
end
U = length(policy.maxU);
if ~exist('m','var'); m = zeros(D,1); end
if ~exist('s','var'); s = 0.001*eye(D); end

% 1. Outputting gradients should not alter non-gradient outputs
[M1,S1,C1] = policy.fcn(policy, m, s);
[M2,S2,C2,~,~,~,~,~,~,~,~,~] = policy.fcn(policy, m, s);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({M1,S1,C1}, {M2,S2,C2})<1e-5, ...
  'policyT: calling derivatives alters non-derivate outputs!')

% 2. TEST OUTPUT-DERIVATIVES --------------------------------------------------

if TEST_DERIVATIVES
  args = {policy, m, s, policy.p, DELTA};
  %[M, S, C,
  %  Mdm, Sdm, Cdm,
  %  Mds, Sds, Cds,
  %  Mdp, Sdp, Cdp] = policy.fcn(policy, m, s, policy.p);
  ntests = 9; test = cell(ntests,1); cg = cell(ntests,1); i = 0;
  i=i+1; test{i} = 'Mdm'; cg{i} = cgwrap(args{:}, [1, 4], 2);
  i=i+1; test{i} = 'Sdm'; cg{i} = cgwrap(args{:}, [2, 5], 2);
  i=i+1; test{i} = 'Cdm'; cg{i} = cgwrap(args{:}, [3, 6], 2);
  i=i+1; test{i} = 'Mds'; cg{i} = cgwrap(args{:}, [1, 7], 3);
  i=i+1; test{i} = 'Sds'; cg{i} = cgwrap(args{:}, [2, 8], 3);
  i=i+1; test{i} = 'Cds'; cg{i} = cgwrap(args{:}, [3, 9], 3);
  i=i+1; test{i} = 'Mdp'; cg{i} = cgwrap(args{:}, [1,10], 4);
  i=i+1; test{i} = 'Sdp'; cg{i} = cgwrap(args{:}, [2,11], 4);
  i=i+1; test{i} = 'Cdp'; cg{i} = cgwrap(args{:}, [3,12], 4);
  print_derivative_test_results(test, cg, EPSILON);
end

% 3. TEST OUTPUTS -------------------------------------------------------------

% Analytic
[Ma, Sa, isCa] = policy.fcn(policy, m, s); Ca = s*isCa;

% Numeric
x = mvnrnd(m,s,N);
z = 0*s;
mn = nan(N,U);
for i=1:N
  print_loop_progress(i,N,'outputs with MC');
  mn(i,:) = policy.fcn(policy, x(i,:)', z);
end
Mn = mean(mn,1); Sn = cov(mn); Cn = nan(D,U);
for d=1:D, for u=1:U, c=cov(x(:,d)',mn(:,u)); Cn(d,u)=c(1,2); end; end

disp('M: mean')
disp('   (analytic)    (numeric)')
disp([Ma Mn]); disp(' ');

disp('S: variance')
disp('   (analytic)    (numeric)')
disp([Sa(:) Sn(:)]); disp(' ');

disp('C: input-output covariance')
disp('   (analytic)    (numeric)')
disp([Ca(:) Cn(:)]); disp(' ');

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
% Example call: dd = cg_wrap(policy,m,s,policy.p,delta,[1,3],2);
function cg = cgwrap(varargin)
[delta, ~, ini] = deal(varargin{end-2:end});
[d,dy,dh] = checkgrad(@cgf,varargin{ini},delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f. Example call:
% dd = checkgrad(@cg_f,policy,m,s,policy.p,delta,[1,3],2);
function [f, df] = cgf(x,varargin)
nout = 3;   % nargout of policy.fcn (non-derivative outputs only)
noutd = 12; % nargout of policy.fcn
nin = 3;    % nargin of policy.fcn
[outi,ini] = deal(varargin{end-1:end}); out=cell(noutd,1);
if ini==3, x=(x+x')/2; end                     % perturb matrices symmetrically
in = varargin(1:nin);
if ini==4;
  in{1}.p = x;
else
  in{ini} = x;
end
policy = varargin{1};
if nargout == 1, [out{1:nout}] = policy.fcn(in{:}); f = out{outi(1)};
else [out{:}] = policy.fcn(in{:}); [f, df] = deal(out{outi}); end