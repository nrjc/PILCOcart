function dcdx = catd(dim,dadx,dbdx,a)
% DCDX = CATD(DIM,DADX,DBDX,A) returns a concatenated derivative-matrix DCDX
% that is the derivative of the matlab function cat(A,B), where DADX and DBDX
% are the derivatives of matrices A and B respectively.
%
% Derivative-matrices are partially unwrapped 4D tensors:
%   1) rows indicated unwrapped dependent-variables,
%   2) columns indicate unwrapped independent-variables.
%
% Example (dim == 1):
%   A              :                  2 x 3  matrix (A is a function of X)
%   B              :                  4 x 3  matrix (B is a function of X)
%   C = cat(1,A,B) :                  6 x 3  matrix
%   X              :                  7 X 8  matrix (independent variable)
%   DADX           : (2*3) x (7*8) =  6 x 56 matrix (derivative of A w.r.t. X)
%   DBDX           : (4*3) x (7*8) = 12 x 56 matrix (derivative of B w.r.t. X)
%   DCDX           : (6*3) x (7*8) = 18 x 56 matrix (derivative of C w.r.t. X)
%
% Example (dim == 2):
%   A              :                  3 x 2  matrix (A is a function of X)
%   B              :                  3 x 4  matrix (B is a function of X)
%   C = cat(2,A,B) :                  3 x 6  matrix
%   X              :                  7 X 8  matrix (independent variable)
%   DADX           : (3*2) x (7*8) =  6 x 56 matrix (derivative of A w.r.t. X)
%   DBDX           : (3*4) x (7*8) = 12 x 56 matrix (derivative of B w.r.t. X)
%   DCDX           : (3*6) x (7*8) = 18 x 56 matrix (derivative of C w.r.t. X)
%
% Currently only handles: dim = 1 or 2.
%
% See also CAT.
% Rowan McAllister 2015-07-19

if dim == 1
  J = size(a,2);
  I1 = size(a,1);
  I2 = I1 + size(dbdx,1) / J;
  nX = size(dbdx,2); % numel(X)
  dcdx = nan(I2*J,nX);
  dcdx(sub2ind2(I2,1:I1,1:J),:) = dadx;
  dcdx(sub2ind2(I2,I1+1:I2,1:J),:) = dbdx;
elseif dim == 2
  dcdx = [dadx ; dbdx];
else
  error('DIM input must be 1 or 2.')
end
