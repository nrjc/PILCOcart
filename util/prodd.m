function dabcdx = prodd(a,dbdx,c)
% DABCDX = PRODD(A,DBDX,C) returns a derivative-matrix DABCDX that is the
% result of each derivative DB (in derivative-matrix DBDX) pre-mutiplied with A
% and post-mutiplied with C. Either A or C inputs are optional (but not both),
% with [] representing a non-input.
%
% Derivative-matrices are partially unwrapped 4D tensors:
% 1) rows indicated unwrapped dependent-variables,
% 2) columns indicate unwrapped independent-variables.
%
% Example:
% A     :                  2 x 3  matrix
% B     :                  3 x 4  matrix (B is a function of X)
% C     :                  4 x 5  matrix
% X     :                  6 X 7  matrix (independent variable)
% DBDX  : (3*4) x (6*7) = 12 x 42 matrix (derivative of  B  w.r.t. X)
% DABCDX: (2*5) x (6*7) = 10 x 42 matrix (derivative of ABC w.r.t. X)
%
% Rowan McAllister 2014-11-18

identity= strcmp(dbdx,'eye'); % faster if dbdx is known to be identity (x == b)

if isempty(a)
  % Case 1: dbdx * c
  if identity
    dabcdx = kron(c',eye(size(c,1)));
  else
    dabcdx = reshape(reshape(dbdx',[],size(c,1))*c,size(dbdx,2),[])';
  end
  
elseif nargin < 3 || isempty(c)
  % Case 2: a * dbdx
  if identity
    dabcdx = kron(eye(size(a,2)),a);
  else
    dabcdx = reshape(a*reshape(dbdx,size(a,2),[]),[],size(dbdx,2));
  end
  
else
  % Case 3: a * dbdx * c
  if identity
    dabcdx = kron(c',a);
  else
    dabcdx = prodd(a,prodd([],dbdx,c));
  end
end