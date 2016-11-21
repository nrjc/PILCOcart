function gpaTh(dyn, m, s, v)

% Test derivatives of the gpa predh function,
%   dyn           dynamics model
%   m     D x 1   mean of state and action
%   s     D x D   variance of state and action
%
% Copyright (C) 2015 Carl Edward Rasmussen and Rowan McAllister 2015-09-24

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
if ~exist('v','var'); v = randn(dyn.D); v = v'*v; end
delta = 1e-4;

douts ={'M', 'S', 'C', 'V'};
dins ={'m', 's', 'v'};
ms = struct('m',m,'s',s,'v',v);

% 2. OUTPUT TEST  -------------------------------------------------------------
% [M1,S1,C1,V1] = dyn.predh(ms.m, ms.s);
% [M2,S2,C2,V2,~,~,~,~,~,~] = dyn.predh(ms.m, ms.s);
% max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
% assert(max_diff({M1,S1,C1,V1},{M2,S2,C2,V2}) < 1e-8, ...
%   'propagateT: requesting derivatives shold not alter non-derivative outputs.');
% assert(isreal([M1(:); S1(:); C1(:); V1(:)]), 'dyn.pred outputs are imaginary!');

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
% [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds] = dyn.predh(self, m, s)
function [f, df] = cg_f(x,varargin)
[dout,din] = deal(varargin{end-1:end});
dyn = varargin{1}; ms = varargin{2};
if any(strcmp(din,{'s','v'})); x = (x+x')/2; end
ms.(din) = x;
if nargout == 1
  [M, S, C, V] = dyn.predh(ms.m, ms.s, ms.v);
else
  [M, S, C, V, dMdm, dSdm, dCdm, dVdm, dMds, dSds, dCds, dVds, dMdv, dSdv, ...
    dCdv, dVdv] = dyn.predh(ms.m, ms.s, ms.v);
  df = eval(['d',dout,'d',din]);
end
f = eval(dout);

