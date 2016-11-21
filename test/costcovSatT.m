function costcovSatT
% Checks the derivatives of the cost function
%
% Edited by Rowan 2015-05-12

clc
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
SEED = 15; rng(SEED);
dbstop if error
DELTA = 1e-4;

% 1. COSTCOVSATT INPUTS -------------------------------------------------------

D = 3;
cost = Cost(D);

i1 = 1:D; i2 = D+1:2*D;
mm = 0.1*randn(2*D,1);
m1 = mm(i1);
m2 = mm(i2);
SS = (1.0)^2*randn(2*D); SS = SS*SS';
S1 = SS(i1,i1);
C = SS(i1,i2);
S2 = SS(i2,i2);

% 2. TEST CTRLBF OUTPUT-DERIVATIVES -------------------------------------------

args = {cost, m1, S1, m2, S2, C};
fncs = {@testm1, @testS1, @testm2, @testS2, @testC};
N = length(fncs);
cg = cell(N,1);
test_names = cell(N,1);
for i = 1:N;
  x = args{i+1};
  [d, dy, dh] = checkgrad(fncs{i}, x, DELTA, args{:});
  cg{i} = {d, dy, dh};
  test_names{i} = func2str(fncs{i});
end
print_derivative_test_results(test_names, cg, EPSILON)

% 3. FUNCTIONS ----------------------------------------------------------------

% varargin = {cost, m1, S1, m2, S2, C}
function [f, df] = testm1(x, varargin)
varargin{2} = x;
[f, df] = nargoutsafe(nargout, varargin{:});

function [f, df] = testS1(x, varargin)
x = (x+x')/2;
varargin{3} = x;
[f, ~, df] = nargoutsafe(nargout, varargin{:});

function [f, df] = testm2(x, varargin)
varargin{4} = x;
[f, ~, ~, df] = nargoutsafe(nargout, varargin{:});

function [f, df] = testS2(x, varargin)
x = (x+x')/2;
varargin{5} = x;
[f, ~, ~, ~, df] = nargoutsafe(nargout, varargin{:});

function [f, df] = testC(x, varargin)
varargin{6} = x;
[f, ~, ~, ~, ~, df] = nargoutsafe(nargout, varargin{:});


function [cov, dcovdm1, dcovdS1, dcovdm2, dcovdS2, dcovdC] = ...
  nargoutsafe(nargout_, varargin)
if nargout_ == 1
  cov = costcovSat(varargin{:});
  [dcovdm1, dcovdS1, dcovdm2, dcovdS2, dcovdC] = deal([],[],[],[],[]);
else
  [cov, dcovdm1, dcovdS1, dcovdm2, dcovdS2, dcovdC] = costcovSat(varargin{:});
end
