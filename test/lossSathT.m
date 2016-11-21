function lossSathT(cost)
% Checks the derivatives of the lossSath function
%
% Edited by Rowan 2016-03-10

clc
%SEED = 2; rng(SEED);
NSAMPLES = 1e4;
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
DELTA = 1e-4;
dbstop if error

% Gradients to test:
dins = {'m', 's', 'v'};
douts = {'M', 'S', 'C', 'V'};

% 1. COST INPUTS --------------------------------------------------------------

if ~exist('cost','var')
  D = 3;
  cost.z = randn(D,1);
  cost.W = randn(D); cost.W = cost.W' * cost.W;
else
  D = length(cost.z);
end

m = randn(D,1);
s = randn(D); s = s'*s;
v = randn(D); v = v'*v;

% 2. TEST CTRLBF OUTPUT-DERIVATIVES -------------------------------------------

ntests = numel(dins)*numel(douts);
test_names = cell(ntests,1); cg = cell(ntests,1); i = 0;
for din = dins;
  for dout = douts;
    i = i+1;
    test_names{i} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{i} = cg_wrap(cost, m, s, v, DELTA, dout{:}, din{:});
  end
end
print_derivative_test_results(test_names, cg, EPSILON)
%return

% 3. TEST CTRLBF OUTPUTS ------------------------------------------------------

% 3.1. Analytic Outputs
[Ma, ~, ~, ~,  Sa, ~, ~, ~, Ca, ~, ~, ~, Va] = lossSath(cost, m, s, v); 
% [M, dMdm, dMds, dMdv,  S, dSdm, dSds, dSdv, C, dCdm, dCds, dCdv, V, dVdm, dVds, dVdv]

% 3.2. Numeric Outputs
mui = mvnrnd(m,s,NSAMPLES);
Mi = nan(NSAMPLES,1);
Si = nan(NSAMPLES,1);
Ci = nan(NSAMPLES,D);
for i = 1:NSAMPLES
  print_loop_progress(i,NSAMPLES, 'Testing outputs with MC');
  [Mi(i),~,~,Si(i),~,~,Ci(i,:)] = lossSat(cost, mui(i,:)', v);
  % [mu, dmudm, dmudS, s2, ds2dm, ds2dS, c, dcdm, dcds]
end
Mn = mean(Mi);
Sn = cov(Mi);
Vn = mean(Si);
Cn = mean(Ci,1);

% Dislpay numeric vs. analytic
str = @(x) (num2str(unwrap(x)'));
fprintf('\nCTRLNF.M MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================\n');
fprintf('M  numeric  : %s\n', str(Mn));
fprintf('M  analytic : %s\n', str(Ma));
fprintf('------------------\n');
fprintf('S  numeric  : %s\n', str(Sn));
fprintf('S  analytic : %s\n', str(Sa));
fprintf('------------------\n');
fprintf('C  numeric  : %s\n', str(Cn));
fprintf('C  analytic : %s\n', str(Ca));
fprintf('------------------\n');
fprintf('V  numeric  : %s\n', str(Vn));
fprintf('V  analytic : %s\n', str(Va));
fprintf('==================\n');

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
% varargin: {s, s2, c, cost, delta, dout, din}
function cg = cg_wrap(varargin)
[cost, m, s, v, delta, dout, din] = deal(varargin{:});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
x = eval(din); % either m, s, or v
[d,dy,dh] = checkgrad(@cg_fcn,x,delta,varargin{:});
cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
% varargin: {s, s2, c, cost, delta, dout, din}
function [f, df] = cg_fcn(x,varargin)
[cost, m, s, v, delta, dout, din] = deal(varargin{:});
switch din
  case 'm'
    m = x;
  case 's'
    s = x; s = (s+s')/2;
  case 'v'
    v = x; v = (v+v')/2;
end
[M, dMdm, dMds, dMdv,  S, dSdm, dSds, dSdv, C, dCdm, dCds, dCdv, ...
  V, dVdm, dVds, dVdv] = lossSath(cost, m, s, v);
f = eval(dout);
df = eval(['d',dout,'d',din]);
