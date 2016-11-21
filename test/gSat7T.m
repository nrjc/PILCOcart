function gSat7T(m, v, i, e)

% Test derivatives of the gSat7 function,
%
% GSAT7T(m, v, i, e):
% m     mean vector of Gaussian                                     [ d       ]
% v     covariance matrix                                           [ d  x  d ]
% i     I length vector of indices of elements to augment
% e     I length optional scale vector (defaults to unity)
%
% Copyright (C) 2015 Carl Edward Rasmussen and Rowan McAllister 2015-07-23

clc
EPSILON = 1e-5;               % 'pass' threshold for low enough checkgrad error
rng(12);

% 1. INPUTS -------------------------------------------------------------------

% default inputs:
if ~exist('m','var'); D = 5; m = randn(D,1); else D = length(m); end
if ~exist('v','var'); v = randn(D); v = v'*v; end
if ~exist('i','var'); U = 2; i = D-U+1:D; else U = length(i); end
if ~exist('e','var'); e = 10*rand(U,1); end
delta = 1e-4;

douts ={'M', 'S', 'C'};
dins ={'m', 'v'};
mv = struct('m',m,'v',v);

% 2. DERIVATIVE TEST  ---------------------------------------------------------

args = {mv, i, e};
ntests = numel(dins)*numel(douts);
test = cell(ntests,1); cg = cell(ntests,1); k = 0;
for din = dins;
  for dout = douts;
    k = k+1;
    test{k} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{k} = cg_wrap(args{:}, delta, dout{:}, din{:});
  end
end
print_derivative_test_results(test, cg, EPSILON)

% 3. OUTPUT TEST  -------------------------------------------------------------
[M1,S1,C1] = gSat7(mv.m, mv.v, i, e);
[M2,S2,C2,~,~,~,~,~,~] = gSat7(mv.m, mv.v, i, e);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({M1,S1,C1},{M2,S2,C2}) < 1e-8, ...
  'propagateT: requesting derivatives shold not alter non-derivative outputs.');

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad,e.g:
% cg_wrap(mv, i, e, delta, dout{:}, din{:})));
function cg = cg_wrap(varargin)
[delta, dout, din] = deal(varargin{end-2:end});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
x = varargin{1}.(din);
[d,dy,dh] = checkgrad(@cg_f,x,delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
% varargin: mv, i, e, delta, dout{:}, din{:}
% [M, S, C, dMdm, dSdm, dCdm, dMdv, dSdv, dCdv] = gSat7(m, v, i, e)
function [f, df] = cg_f(x,varargin)
[dout,din] = deal(varargin{end-1:end});
mv = varargin{1}; i = varargin{2}; e = varargin{3};
if strcmp(din,'v'); x = (x+x')/2; end
mv.(din) = x;
if nargout == 1
  [M, S, C] = gSat7(mv.m, mv.v, i, e);
else
  [M, S, C, dMdm, dSdm, dCdm, dMdv, dSdv, dCdv] = gSat7(mv.m, mv.v, i, e);
  switch din
    case 'm'
      switch dout
        case 'M'
          df = dMdm;
        case 'S'
          df = dSdm;
        case 'C'
          df = dCdm;
      end
    case 'v'
      switch dout
        case 'M'
          df = dMdv;
        case 'S'
          df = dSdv;
        case 'C'
          df = dCdv;
      end
  end
end
switch dout
  case 'M'
    f = M;
  case 'S'
    f = S;
  case 'C'
    f = C;
end