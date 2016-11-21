function gpaT2(dyn, m, s)

% Test derivatives of the gpa pred function,
%   dyn           dynamics model
%   m     D x 1   mean of state and action
%   s     D x D   variance of state and action
%
% Copyright (C) 2015 Carl Edward Rasmussen and Rowan McAllister 2015-07-23

clc
EXPERIMENT = 'unicycle';
EPSILON = 1e-5;               % 'pass' threshold for low enough checkgrad error
rng(12);

% 1. INPUTS -------------------------------------------------------------------

% default inputs:
plant = create_test_object('plant', EXPERIMENT);
if ~exist('dyn','var'); dyn = create_test_object('dyn', plant); end
if ~exist('m','var'); m = randn(dyn.D,1); end
if ~exist('s','var'); s = randn(dyn.D); s = s'*s; end
delta = 1e-4;

douts ={'M', 'S', 'C'};
dins ={'m', 's'};
douts = {'M','S'}; % override
dins = {'m'}; % override
ms = struct('m',m,'s',s);

% 2. OUTPUT TEST  -------------------------------------------------------------
[M1,S1,C1] = dyn.pred(ms.m, ms.s);
[M2,S2,C2,~,~,~,~,~,~] = dyn.pred(ms.m, ms.s);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({M1,S1,C1},{M2,S2,C2}) < 1e-8, ...
  'propagateT: requesting derivatives shold not alter non-derivative outputs.');
assert(isreal([M1(:); S1(:); C1(:)]), 'dyn.pred outputs are imaginary!');

% 3. DERIVATIVE TEST  ---------------------------------------------------------

args = {dyn, ms};
ntests = numel(dins)*numel(douts);
test = cell(ntests,1); cg = cell(ntests,1); i = 0;
for din = dins;
  for dout = douts;
    i = i+1;
    test{i} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{i} = cg_wrap(args{:}, delta, dout{:}, din{:});
  end
end
print_derivative_test_results(test, cg, EPSILON)

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad,e.g:
% cg_wrap(dyn, ms, delta, dout{:}, din{:})));
function cg = cg_wrap(varargin)
[delta, dout, din] = deal(varargin{end-2:end});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
x = varargin{2}.(din);
[d,dy,dh] = checkgrad(@cg_f,x,delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
% varargin: dyn, ms, delta, dout{:}, din{:}
% [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds] = pred(self, m, s)
function [f, df] = cg_f(x,varargin)
[dout,din] = deal(varargin{end-1:end});
dyn = varargin{1}; ms = varargin{2};
if strcmp(din,'s'); x = (x+x')/2; end
ms.(din) = x;
if nargout == 1
  [M, S, C] = gpp(dyn,ms.m, ms.s);
else
  [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds] = gpd(dyn,ms.m, ms.s);
  df = eval(['d',dout,'d',din]);
end
f = eval(dout);
