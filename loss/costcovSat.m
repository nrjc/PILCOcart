function [cov, dcovdm1, dcovdS1, dcovdm2, dcovdS2, dcovdC] = ...
  costcovSat(cost, m1, S1, m2, S2, C)
% Compute covariance of the saturating costs
%   1 - exp(-(x-z)'*W*(x-z)/2) and
%   1 - exp(-(y-z)'*W*(y-z)/2) and
%   derivatives, where x ~ N(m1,S1), y ~ N(m2,S2)
%
% [cov, dcovdm1, dcovdS1, dcovdm2, dcovdS2, dcovdC] = ...
%                                           costcovSat(cost, m1, S1, m2, S2, C)
%
% cost     .       cost structure
%   z      D x 1   target state (default: zero vector)
%   W      D x D   weight matrix (default: identity matrix)
% m1       D x 1   mean of the state 1 distribution
% S1       D x D   covariance matrix of the state 1 distribution
% m2       D x 1   mean of the state 2 distribution
% S2       D x D   covariance matrix of the state 2 distribution
% C        D x D   covariance of state 1 and state 2
% cov              cost covariance (due to covariances between states)
% dcovdm1  1 x D   derivative of cost covariance wrt state 1 mean vector
% dcovdS1  1 x DD  derivative of cost covariance wrt state 1 covariance matrix
% dcovdm2  1 x D   derivative of cost covariance wrt state 2 mean vector
% dcovdS2  1 x DD  derivative of cost covariance wrt state 2 variance matrix
% dcovdC   1 x DD  derivative of cost covariance wrt states covariance matrix
%
% Copyright (C) 2008-2015 by Carl Edward Rasmussen & Rowan McAllister
% 2012-05-12

D = length(m1);                                           % get state dimension
if isfield(cost,'W'); W = cost.W; else W = eye(D); end        % set defaults ..
if isfield(cost,'z'); z = cost.z; else z = zeros(D,1); end    % .. if necessary
derivatives_requested = nargout > 1;

zz = [z ; z];
WW = blkdiag(W, W);

m = [m1 ; m2];
S = [S1 , C ; C', S2];

if ~derivatives_requested
  mu = lossSat(struct('z',zz,'W',WW), m, S);
  mu1 = lossSat(struct('z',z,'W',W), m1, S1);
  mu2 = lossSat(struct('z',z,'W',W), m2, S2);
else
  [mu, dmudm, dmudS] = lossSat(struct('z',zz,'W',WW), m, S);
  [mu1, dmu1dm1, dmu1dS1] = lossSat(struct('z',z,'W',W), m1, S1);
  [mu2, dmu1dm2, dmu2dS2] = lossSat(struct('z',z,'W',W), m2, S2);
  i1 = 1:D; i2 = D+1:2*D;
  dcovdm1 = unwrap(-dmudm(i1)    + dmu1dm1*(1-mu2))';
  dcovdS1 = unwrap(-dmudS(i1,i1) + dmu1dS1*(1-mu2))';
  dcovdm2 = unwrap(-dmudm(i2)    + (1-mu1)*dmu1dm2)';
  dcovdS2 = unwrap(-dmudS(i2,i2) + (1-mu1)*dmu2dS2)';
  dcovdC = unwrap(-dmudS(i1,i2)-dmudS(i2,i1)')';
end
cov = (1-mu) - (1-mu1)*(1-mu2);
