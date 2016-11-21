function [M, dMdm, dMds, dMdv,  S, dSdm, dSds, dSdv, C, dCdm, dCds, dCdv, ...
  V, dVdm, dVds, dVdv] = lossSath(cost, m, s, v)
% Compute expectation and variance of the saturating cost
% 1 - exp(-(x-z)'*W*(x-z)/2) and its derivatives, where x ~ N(N(m,s),v).
%
% [M, dMdm, dMds, dMdv,  S, dSdm, dSds, dSdv, C, dCdm, dCds, dCdv, ...
%   V, dVdm, dVds, dVdv] = lossSath(cost, m, s, v)
%
% m      D x 1  mean-of-mean of the state distribution
% s      D x D  variance-of-mean matrix of the state distribution
% v      D x D  mean-of-variance matrix of the state distribution
% cost   .      cost structure
%   z    D x 1  target state (default: zero vector)
%   W    D x D  weight matrix (default: identity matrix)
% M      1 x 1  mean-of-mean of cost
% dMdm   1 x D  derivative of M wrt m
% dMds   1 xDD  derivative of M wrt s
% dMdv   1 xDD  derivative of M wrt v
% S      1 x 1  variance-of-mean of cost
% dSdm   1 x D  derivative of S wrt m
% dSds   1 xDD  derivative of S wrt s
% dSdv   1 xDD  derivative of S wrt v
% C      D x 1  inv(s) times input-output covariance
% dCdm   D x D  derivative of C wrt m
% dCds   D xDD  derivative of C wrt s
% dCdv   D xDD  derivative of C wrt v
% V      1 x 1  mean-of-variance of cost
% dVdm   1 x D  derivative of V wrt m
% dVds   1 xDD  derivative of V wrt s
% dVdv   1 xDD  derivative of V wrt v
%
% See also LOSSSAT.M and <a href="cost.pdf">cost.pdf</a>
% Copyright (C) 2016 by Carl Edward Rasmussen and Rowan McAllister 2016-03-10

D = length(m);                                             % get state dimension
I = eye(D);
if isfield(cost,'W'); W = cost.W; else W = eye(D); end         % set defaults ..

% compute M and C:
[M, dMdm, dMds, S1, dS1dm, dS1dsv, C, dCdm, dCds] = lossSat(cost, m, s + v);
dMds = dMds(:)';
dMdv = dMds;
dS1dsv = dS1dsv(:)';
dCdv = dCds;

% compute S and V:
[M2, dM2dm, dM2ds, S2, dS2dm, dS2ds] = lossSat(cost, m, (2*s + v)/2);
dM2ds = dM2ds(:)';
dS2ds = dS2ds(:)';
dM3dv = dM2ds/2;
dS3dv = dS2ds/2;

cS = 1/sqrt(det(I+v*W));
S = cS * (S2 + (M2-1)^2) - (M-1)^2;
dcSdv = - cS * W/(I+v*W)/2; dcSdv = dcSdv(:)';
dSdm = cS * (dS2dm + 2*(M2-1)*dM2dm) - 2*(M-1)*dMdm;
dSds = cS * (dS2ds + 2*(M2-1)*dM2ds) - 2*(M-1)*dMds;
dSdv = dcSdv * (S2 + (M2-1)^2) + cS * (dS3dv + 2*(M2-1)*dM3dv) - 2*(M-1)*dMdv;

V = S1 - S;
dVdm = dS1dm - dSdm;
dVds = dS1dsv - dSds;
dVdv = dS1dsv - dSdv;

% A cleaner way to compute S, but I cannot get dSdv gradients this way yet.
%
% W1 = W/(I + v*W);
% W2 = 2*W/(I + 2*v*W);
% cost1 = struct('z',z, 'W',W1);
% cost2 = struct('z',z, 'W',W2);
% if isfield(cost,'z'); z = cost.z; else z = zeros(D,1); end     % .. if necessary
%
% [~, ~, ~, S1, dS1dm, dS1ds] = lossSat(cost1, m, s);
% dS1ds = dS1ds(:)';
% cS = 1/det(I+v*W);
% S = cS * S1;
% dSdm = cS * dS1dm;
% dSds = cS * dS1ds;
