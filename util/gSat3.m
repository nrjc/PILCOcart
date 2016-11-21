% Compute moments of the saturating function e*(9*sin(2*x(i)/3)+sin(2*x(i)))/8, 
% where x sim N(m,v) and i is a (possibly empty) set of I indices. The optional
% e scaling factor is a vector of length I. Optionally, compute derivatives of
% the moments.
%
% Copyright (C) 2015 by Carl Edward Rasmussen, Andrew McHutchon and
%                                                    Marc Deisenroth 2015-03-12

function [M, S, C, dMdm, dSdm, dCdm, dMdv, dSdv, dCdv] = gSat3(m, v, i, e)

% m     mean vector of Gaussian                                     [ d       ]
% v     covariance matrix                                           [ d  x  d ]
% i     I length vector of indices of elements to augment
% e     I length optional scale vector (defaults to unity)
%
% M     output means                                                [ I       ]
% V     output covariance matrix                                    [ I  x  I ]
% C     inv(v) times input-output covariance                        [ d  x  I ]
% dMdm  derivatives of M w.r.t m                                    [ I  x  d ]
% dVdm  derivatives of V w.r.t m                                    [I*I x  d ]
% dCdm  derivatives of C w.r.t m                                    [d*I x  d ]
% dMdv  derivatives of M w.r.t v                                    [ I  x d*d]
% dVdv  derivatives of V w.r.t v                                    [I*I x d*d]
% dCdv  derivatives of C w.r.t v                                    [d*I x d*d]

d = length(m); I = length(i); i = i(:)'; f = [2/3; 2]; F = length(f); 
if nargin < 4; e = ones(1, I); end; e = e(:)';

P = eye(d+F*I,d); P(d+1:d+F*I,i) = kron(eye(I),f);             % augment inputs
ma = P*m;    madm = P;
va = P*v*P'; vadv = kron(P,P); va = (va+va')/2;

[M2, S2, C2, Mdma, Sdma, Cdma, Mdva, Sdva, Cdva] ...
                                          = gSin(ma, va, d+1:d+F*I, [9*e e]/8);

R = [eye(I) eye(I)];
M = R*M2;                                                                % mean
S = R*S2*R'; S = (S+S')/2;                                           % variance
C = P'*C2*R';                                   % inv(v) times input-output cov

if nargout > 3                                        % derivatives if required
  dMdm = R*Mdma*madm;          dMdv = R*Mdva*vadv;
  dSdm = kron(R,R)*Sdma*madm;  dSdv = kron(R,R)*Sdva*vadv;
  dCdm = kron(R,P')*Cdma*madm; dCdv = kron(R,P')*Cdva*vadv;
end
