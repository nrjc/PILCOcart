function [L, S2, dLd] = lossMK(cost, s)

% Cart-pole loss function. The loss is 1 - exp(-0.5*a*d^2), where "a" is a
% (positive) constant and "d^2" is the squared (Euclidean) distance between
% the tip of the pendulum and the upright position.
%
% If the exploration parameter "b" is not present (or if it is zero) then
% the expected loss is computed, averaging over the Gaussian distribution
% of the state, with mean "mu" and covariance matrix "Sigma". If it is
% present, then sum of the average loss and "b" times the std. deviation
% of the loss is returned. Negative values of "b" are used to encourage
% exploration and positive values avoid regions of uncertainty in the
% policy. Derivatives of these quantities are computed when desired. See
% also task.pdf.
%
% s        .    state structure
%   m      5x1  mean of state distribution
%   s      5x5  covariance matrix for the state distribution
% cost     .    cost structure
%   ell         length of the pendulum
%   width       array of widths of the cost (summed together)
%   expl        exploration parameter
% L             expected cost
% S2
% dLd           derivative of expected cost wrt state structure
%
% Copyright (C) 2008-2015 by Carl Edward Rasmussen 2015-01-14

if isfield(cost,'expl'), b = cost.expl; else b = 0; end
D = length(s.m); if ~isfield(s,'s'), s.s = zeros(D); end

[M, S, C, Mdm, Sdm, Cdm, Mds, Sds, Cds] = gTrigN(s.m, s.s, D-0);
  
L = 0; dLd.m = zeros(1,D); dLd.s = zeros(1,D*D); S2 = 0;
cost.z(D+2,1) = 1;
Q(D+2,D+2) = cost.ell^2; Q([D-1 D+1],[D-1 D+1]) = [1 -cost.ell]'*[1 -cost.ell];
for i = 1:length(cost.width)                 % scale mixture of immediate costs
  cost.W = Q/cost.width(i)^2;
  [r, rdM, rdS, s2, s2dM, s2dS] = lossSat(cost, M, S);
  L = L + r; S2 = S2 + s2;
  if nargout > 2
    dLd.m = dLd.m + rdM*Mdm + rdS(:)'*Sdm;
    dLd.s = dLd.s + rdM*Mds + rdS(:)'*Sds;
  end
  if (b~=0 || ~isempty(b)) && abs(s2)>1e-12
    L = L + b*sqrt(s2);
    if nargout > 2
      dLd.m = dLd.m + b/sqrt(s2) * (s2dM(:)'*Mdm + s2dS(:)'*Sdm)/2;
      dLd.s = dLd.s + b/sqrt(s2) * (s2dM(:)'*Mds + s2dS(:)'*Sds)/2;
    end
  end
end

n = length(cost.width);                                             % normalize
L = L/n;  S2 = S2/n;
if nargout > 2
  dLd.m = dLd.m/n;
  dLd.s = reshape(dLd.s/n,D,D);
end;
