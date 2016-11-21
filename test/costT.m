function costT(cost, s1, s2, delta)
% Checks the derivatives of the cost function
%
% Edited by Rowan 2016-03-15

clc
% SEED = 19; rng(SEED);
NSAMPLES = 1e4;
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
dbstop if error
EXPERIMENT = 'cartPole'; % 'cartPole' or 'cartDoublePendulum' or 'unicycle'
EXPERIMENT_SUFFIX = ''; % '' or 'Markov'

% Gradients to test:
douts ={'m', 's', 'c', 'cov'};
dins ={'1m', '1s', '2m', '2s', '0c'};

% Hierarchical Gradients to test:
douths ={'m', 's', 'c', 'v'};
dinhs ={'m', 's', 'v'};

max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
str = @(x) (num2str(unwrap(x)'));

% 1. COST INPUTS --------------------------------------------------------------

% cd(['/homes/mlghomes/rtm26/ctrl/scenarios/', EXPERIMENT])
if exist('cost','var');
  D = cost.D;
else plant = create_test_object('plant', [EXPERIMENT, EXPERIMENT_SUFFIX]);
  D = plant.D;
  cost = Cost(D);
end

i1 = 1:D; i2 = D+1:2*D;
if ~exist('s2','var')
  disp('costT: overwriting s1')
  % mm = 0.05*randn(2*D,1);
  mm = 0.72*ones(2*D,1); % tmp
  s1.m = mm(i1);
  s2.m = mm(i2);
  ss = (0.1*randn(2*D))^2; ss = ss*ss';
  s1.s = ss(i1,i1);
  c = ss(i1,i2);
  s2.s = ss(i2,i2);
  % only for the hierarchical test_names
  sh = s1;
  sh.v = (0.1*randn(D))^2; sh.v = sh.v'*sh.v;
end
if ~exist('delta','var'); delta = 1e-4; end

% 2. TEST COSTT OUTPUT-DERIVATIVES --------------------------------------------

ntests = numel(dins)*numel(douts);
test_names = cell(ntests,1); cg = cell(ntests,1); i = 0;
for din = dins;
  for dout = douts;
    if ~strcmp(dout,'cov') && din{:}(1) ~= '1'; continue; end % not meaningful
    i = i+1;
    test_names{i} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{i} = cg_wrap(s1, s2, c, cost, delta, dout{:}, din{:});
  end
end
test_names = test_names(1:i); cg = cg(1:i); % trim

% 3. TEST COSTT HIERACHICAL-OUTPUT-DERIVATIVES --------------------------------

ntests = numel(dinhs)*numel(douths);
testh_names = cell(ntests,1); cgh = cell(ntests,1); i = 0;
for dinh = dinhs;
  for douth = douths;
    i = i+1;
    testh_names{i} = strcat('d(',douth{:},')/d(',dinh{:},'h): ');
    cgh{i} = cgh_wrap(sh, cost, delta, douth{:}, dinh{:});
  end
end
testh_names = testh_names(1:i); cgh = cgh(1:i); % trim

% 3b. Print all derrivative results -------------------------------------------
print_derivative_test_results(test_names, cg, EPSILON)
print_derivative_test_results(testh_names, cgh, EPSILON)

% 4. TEST COSTT OUTPUTS -------------------------------------------------------

% 4.1. Analytic Outputs
L1 = cost.fcn(s1);
[L1_,~] = cost.fcn(s1);
assert(max_diff(L1,L1_) < 1e-10, ...
  'lossT: requesting derivatives shold not alter non-derivative outputs.');
[cova,~] = cost.cov(s1, s2, c);
[cova_, ~, ~, ~] = cost.cov(s1, s2, c);
assert(max_diff(cova,cova_) < 1e-10, ...
  'lossT: requesting derivatives shold not alter non-derivative outputs.');
L2 = cost.fcn(s2);
L1Cov = s1.s*L1.c;
L2Cov = s2.s*L2.c;

% 4.2. Numeric Outputs
x = mvnrnd(mm,ss,NSAMPLES);
L1i = nan(NSAMPLES,1);
L2i = nan(NSAMPLES,1);
for i = 1:NSAMPLES
  print_loop_progress(i,NSAMPLES, 'Testing outputs with MC');
  s1i.m = x(i,i1)';
  s2i.m = x(i,i2)';
  L1i_ = cost.fcn(s1i); L1i(i,:) = L1i_.m; assert(L1i_.s == 0);
  L2i_ = cost.fcn(s2i); L2i(i,:) = L2i_.m; assert(L2i_.s == 0);
end
L1Mn = mean(L1i);
L2Mn = mean(L2i);
L1Sn = cov(L1i);
L2Sn = cov(L2i);
L1Cn = nan(D,1);
L2Cn = nan(D,1);
for i = 1:length(i1); c = cov(x(:,i1(i)),L1i); L1Cn(i) = c(1,2); end
for i = 1:length(i2); c = cov(x(:,i2(i)),L2i); L2Cn(i) = c(1,2); end
covn = cov(L1i,L2i); covn = covn(1,2);

% 4.3. Dislpay numeric vs. analytic
fprintf('\nCOSTT.M NON-HIERARCHICAL MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================\n');
fprintf('L1M  numeric  : %s\n', str(L1Mn));
fprintf('L1M  analytic : %s\n', str(L1.m));
fprintf('------------------\n');
fprintf('L2M  numeric  : %s\n', str(L2Mn));
fprintf('L2M  analytic : %s\n', str(L2.m));
fprintf('------------------\n');
fprintf('L1S  numeric  : %s\n', str(L1Sn));
fprintf('L1S  analytic : %s\n', str(L1.s));
fprintf('------------------\n');
fprintf('L2S  numeric  : %s\n', str(L2Sn));
fprintf('L2S  analytic : %s\n', str(L2.s));
fprintf('------------------\n');
fprintf('L1C  numeric  : %s\n', str(L1Cn));
fprintf('L1C  analytic : %s\n', str(L1Cov));
fprintf('------------------\n');
fprintf('L2C  numeric  : %s\n', str(L2Cn));
fprintf('L2C  analytic : %s\n', str(L2Cov));
fprintf('------------------\n');
fprintf('cov  numeric  : %s\n', str(covn));
fprintf('cov  analytic : %s\n', str(cova));
fprintf('==================\n');
mdiff = max_diff({L1Mn,L2Mn,L1Sn,L2Sn,covn}, {L1.m,L2.m,L1.s,L2.s,cova});
fprintf('Maximum difference = %4.2e\n', mdiff);

% 5. TEST COSTT HIERACHICAL OUTPUTS -------------------------------------------

% 5.1. Analytic Outputs
Lh = cost.fcnh(sh);
[Lh_,~] = cost.fcnh(sh);
assert(max_diff(Lh,Lh_) < 1e-10, ...
  'lossT: requesting derivatives shold not alter non-derivative outputs.');
%LhCov = sh.s*Lh.c;

% 5.2. Numeric Outputs
x = mvnrnd(sh.m,sh.s,NSAMPLES);
Lhmi = nan(NSAMPLES,1);
Lhsi = nan(NSAMPLES,1);
si.s = sh.v;
for i = 1:NSAMPLES
  print_loop_progress(i,NSAMPLES, 'Testing hierarchical-outputs with MC');
  si.m = x(i,:)';
  Lhi_ = cost.fcn(si); Lhmi(i,:) = Lhi_.m; Lhsi(i,:) = Lhi_.s;
end
LhMn = mean(Lhmi);
LhSn = cov(Lhmi);
LhVn = mean(Lhsi);
LhCn = nan(D,1);
for i = 1:D; c = cov(x(:,i),Lhmi); LhCn(i) = c(1,2); end

% 5.3. Dislpay numeric vs. analytic
fprintf('\nCOSTT.M HIERARCHICAL MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================\n');
fprintf('LhM  numeric  : %s\n', str(LhMn));
fprintf('LhM  analytic : %s\n', str(Lh.m));
fprintf('------------------\n');
fprintf('LhS  numeric  : %s\n', str(LhSn));
fprintf('LhS  analytic : %s\n', str(Lh.s));
fprintf('------------------\n');
fprintf('LhC  numeric  : %s\n', str(LhCn));
fprintf('LhC  analytic : %s\n', str(sh.s*Lh.c));
fprintf('------------------\n');
fprintf('LhV  numeric  : %s\n', str(LhVn));
fprintf('LhV  analytic : %s\n', str(Lh.v));
fprintf('==================\n');
mdiff = max_diff({LhMn,LhSn,LhVn}, {Lh.m,Lh.s,Lh.v});
fprintf('Maximum difference = %4.2e\n', mdiff);

% 6. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
function cg = cg_wrap(varargin)
[s1, s2, c, cost, delta, dout, din] = deal(varargin{:});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
s_all = {s1,s2};
k = str2double(din(1));
din = din(2);
if strcmp(din, 'c')
  x = c;
else
  x = s_all{k}.(din);
end
if strcmp(dout, 'cov')
  [d,dy,dh] = checkgrad(@cg_cov,x,delta,varargin{:});
else
  [d,dy,dh] = checkgrad(@cg_L,x,delta,varargin{:});
end
cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
function [f, df] = cg_L(x,varargin)
[s, s2, c, cost, delta, dout, din] = deal(varargin{:});
D = cost.D;
din = din(2);
if strcmp(din,'s'); x = (x+x')/2; end
s.(din) = x;
if nargout == 1
  L = cost.fcn(s);
else % compute derivatives
  [L, dL] = cost.fcn(s);
  if din == 'm', idin = 1:D; elseif din == 's', idin = D+1:D+D*D; end
  df = dL.(dout)(:,idin);
end
f = L.(dout);

% Checkgrad input function (covariance) version.
function [f, df] = cg_cov(x,varargin)
[s1, s2, c, cost, delta, dout, din] = deal(varargin{:});
D = cost.D;
s_all = {s1,s2};
k = str2double(din(1));
din = din(2);
if din == 'c'
  c = x;
else
  if strcmp(din,{'s'}); x = (x+x')/2; end
  s_all{k}.(din) = x;
end
[s1, s2] = deal(s_all{:});
if nargout == 1
  cov = cost.cov(s1, s2, c);
else % compute derivatives
  [cov, ds1, ds2, dc] = cost.cov(s1, s2, c);
  if din == 'c'
    df = dc;
  else
    ds_all = {ds1, ds2};
    if din == 'm', j = 1:D; elseif din == 's', j = D+1:D+D*D; end
    df = ds_all{k}(j);
  end
end
f = cov;

% Checkgrad wrapper (hierarchical) version.
function cg = cgh_wrap(varargin)
[s, cost, delta, dout, din] = deal(varargin{:});
disp([mfilename,': derivative test: d(',dout,')/d(',din,'h)']);
x = s.(din);
[d,dy,dh] = checkgrad(@cg_Lh,x,delta,varargin{:});
cg = {d dy dh};

% Checkgrad input function (hierarchical) version.
function [f, df] = cg_Lh(x,varargin)
[s, cost, delta, dout, din] = deal(varargin{:});
D = cost.D;
if any(strcmp(din,{'s','v'})); x = (x+x')/2; end
s.(din) = x;
if nargout == 1
  L = cost.fcnh(s);
else % compute derivatives
  [L, dL] = cost.fcnh(s);
  if din == 'm', idin = 1:D;                                                    % TODO fix this?
  elseif din == 's', idin = D+1:D+D*D;
  elseif din == 'v', idin = D+D*D+1:D+2*D*D;
  end
  df = dL.(dout)(:,idin);
end
f = L.(dout);