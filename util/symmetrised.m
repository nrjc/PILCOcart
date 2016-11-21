function dx = symmetrised(dx,dim)
% DX = SYMMETRISED(DX,DIM) symeterises part of a (D)erivative-matrix DX whose
% rows represent upwrapped dependent variables, and columns represent unwrapped
% independent variables. Either the dependent variable can be symeterised
% (first dimension of DX) by setting DIM=1, or the independent variable may be
% symeterised (second dimension of DX) by setting DIM=2.
%
% dx = symmetrised(dx,dim)
%
% dx   A*A x B*B   derivative-matrix
% dim              dimension(s) to symeterise, 1 or 2 or both [1,2]
%
% Rowan McAllister 2014-12-01

if any(dim == 1)
  % symeterise the dependent variable
  A = sqrt(size(dx,1));
  if(A ~= floor(A)); error('derivative matrix numerator is non-square.'); end
  dx = (dx + transposed(dx,A,1))/2;
end
if any(dim == 2)
  % symeterise the independent variable
  B = sqrt(size(dx,2));
  if(B ~= floor(B)); error('derivative matrix denominator is non-square.'); end
  dx = (dx + transposed(dx,B,2))/2;
end