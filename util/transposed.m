function xt = transposed(x,s,dim)
% XT = TRANSPOSED(X,S,DIM) takes a transpose of a derivative-matrix whose rows
% represent upwrapped dependent variables, and columns represent unwrapped
% independent variables. Either the dependent variable can be transposed
% (first dimension of X) by setting DIM=1, or the independent variable may be
% transposed (second dimension of X) by setting DIM=2.
%
% xt = transposed(x,s,dim)
%
%   x   A*B x C*D   derivative-matrix
% CASE DIM == 1:
%   s               s = A, i.e. size of first dim of unwrapped matrix AB
%   xt  B*A x C*D   transpose of the dependent variable
% CASE DIM == 2:
%   s               s = C, i.e. size of first dim of unwrapped matrix CD
%   xt  A*B x D*C   transpose of the independent variable
%
% Note: s can also be the dependent or independent matrix variable, in which
% case it's size will be used to determine length A and B, or C and D.
%
% Rowan McAllister 2015-05-15

if numel(x) == 0; xt = x; return; end
if nargin < 3; dim = 1; end                                     % default value
if dim == 1
  % transpose of dependent variable
  if numel(s) > 1
    A = size(s,1);
    B = size(s,2);
  else
    A = s;
    B = size(x,1)/A;
  end
  xt = reshape(permute(reshape(x,A,B,[]),[2 1 3]),A*B,[]);
elseif dim == 2
  % transpose of independent variable
  if numel(s) > 1
    C = size(s,1);
    D = size(s,2);
  else
    C = s;
    D = size(x,2)/C;
  end
  xt = reshape(permute(reshape(x,[],C,D),[1 3 2]),[],C*D);
end