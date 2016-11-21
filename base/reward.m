function [mu, dmudm, dmudS, s2, ds2dm, ds2dS] = reward(m, S, z, W)

% Compute expectation and variance of a quadratic cost and their derivatives
%
% input arguments:
% m:  D-by-1 mean of the state distribution
% S:  D-by-D covariance matrix of the state distribution
% z:  D-by-1 target state
% W: D-by-D weight matrix
%
% output arguments:
% muR:        1-by-1 expected reward
% dmuRdm:     1-by-D derivative of expected reward wrt input mean
% dmuRdS:     D-by-D derivative of expected reward wrt input covariance matrix
% s2R:        1-by-1 variance of reward
% ds2Rdm:     1-by-D derivative of variance of reward wrt input mean
% ds2RdS:     D-by-D derivative of variance of reward wrt input covariance
% matrix
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen and Marc Deisenroth
% 2012-06-19

% some precomputations
D = length(m); % get state dimension
SW = S*W;
iSpW = W/(eye(D)+SW);

mu = exp(-(m-z)'*iSpW*(m-z)/2)/sqrt(det(eye(D)+SW));  % expected reward

% dervivatives of expected reward
if nargout > 1
  dmudm = -mu*(m-z)'*iSpW;  % wrt input mean
  dmudS = mu*(iSpW*(m-z)*(m-z)'-eye(D))*iSpW/2;  % wrt input covariance matrix
end

% variance of reward
if nargout > 3
  i2SpW = W/(eye(D)+2*SW);
  r2 = exp(-(m-z)'*i2SpW*(m-z))/sqrt(det(eye(D)+2*SW));
  s2 = r2 - mu^2;
  if s2 < 1e-12; s2=0; end % for numerical reasons
end

% derivatives of variance of reward
if nargout > 4
  % wrt input mean
  ds2dm = -2*r2*(m-z)'*i2SpW-2*mu*dmudm;
  % wrt input covariance matrix
  ds2dS = r2*(2*i2SpW*(m-z)*(m-z)'-eye(D))*i2SpW-2*mu*dmudS;
end

