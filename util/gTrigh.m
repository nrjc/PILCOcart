function [M, S, C, V, ...
  dMdm, dSdm, dCdm, dVdm, ...
  dMds, dSds, dCds, dVds, ...
  dMdv, dSdv, dCdv, dVdv] = gTrigh(m, s, v, i, e)

% Compute moments of trigonometric functions e*sin(x(i)) and e*cos(x(i)), where
% x sim N(N(m,s),v) and i is a (possibly empty) set of I indices. The optional
% e scaling factor is a vector of length I. Optionally, compute derivatives of
% the moments.
%
% m     mean-of-mean Gaussian vector                                [ d  x  1 ]
% s     mean-covariance matrix                                      [ d  x  d ]
% v     covariance matrix                                           [ d  x  d ]
% i     I length vector of indices of elements to augment
% e     I length optional scale vector (defaults to unity)
%
% M     output mean-of-mean                                         [ 2I      ]
% S     output covariance-of-mean matrix                            [ 2I x 2I ]
% C     inv(s) times mean input - mean output cov, equivalently     [ d  x 2I ]
%       inv(v) times expected[input-output covariance]
% V     output mean-of-covariance matrix                            [ 2I x 2I ]
% dMdm  derivative of M wrt m                                       [ 2I x  d ]
% dSdm  derivative of S wrt m                                       [4II x  d ]
% dCdm  derivative of C wrt m                                       [2dI x  d ]
% dVdm  derivative of V wrt m                                       [4II x  d ]
% dMds  derivative of M wrt s                                       [ 2I x dd ]
% dSds  derivative of S wrt s                                       [4II x dd ]
% dCds  derivative of C wrt s                                       [2dI x dd ]
% dVds  derivative of V wrt s                                       [4II x dd ]
% dMdv  derivative of M wrt v                                       [ 2I x dd ]
% dSdv  derivative of S wrt v                                       [4II x dd ]
% dCdv  derivative of C wrt v                                       [2dI x dd ]
% dVdv  derivative of V wrt v                                       [4II x dd ]
%
% See also <a href="trighaug.pdf">trighaug.pdf</a>, GTRIG.M.
% Copyright (C) 2014 by Carl Edward Rasmussen, Rowan McAllister 2014-12-09

I = length(i); Ic = 2*(1:I); Is = Ic-1;
if nargin == 4; e = ones(I,1); end
vi = diag(v(i,i)); vii = nan(2*I,1); vii(Ic) = vi; vii(Is) = vi;
q = exp(-bsxfun(@plus,vii,vii')/2);

if nargout <= 4
  [~,S] = gTrig(m,s,i,e); S = q.*S;
  [M,V,C] = gTrig(m,s+v,i,e); V = V-S;
  return
end                                                  % else compute derivatives

[~,S,~,~,dSdm,~,~,dSds] = gTrig(m,s,i,e); S = q.*S;
[M,V,C,dMdm,dVdm,dCdm,dMdv,dVdv,dCdv] = gTrig(m,s+v,i,e); V = V-S;

q = q(:); D = length(m);
dSds = symmetrised(dSds,[1,2]);
dVdv = symmetrised(dVdv,[1,2]);
dSdm = bsxfun(@times,q,dSdm);
dSds = bsxfun(@times,q,dSds);
dSdv = bsxfun(@times,dlqdv(D,i),S(:));
dMds = dMdv;
dCds = dCdv;
dVdm = dVdm-dSdm;
dVds = dVdv-dSds;
dVdv = dVdv-dSdv;


function dlq = dlqdv(D,i)
I = length(i);
dlq = zeros(4*I*I,D*D);
for k = 1:length(i);
  m = zeros(2*I); m([2*k-1,2*k],:)=-0.5; m=m+m';
  dlq(:,sub2ind2(D,i(k),i(k))) = m(:);
end