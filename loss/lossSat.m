function [mu, dmudm, dmudS, s2, ds2dm, ds2dS, c, dcdm, dcds] = ...
                                                             lossSat(cost, m, S)
% Compute expectation and variance of the saturating cost
% 1 - exp(-(x-z)'*W*(x-z)/2) and its derivatives, where x ~ N(m,S).
%
% m      D x 1  mean of the state distribution
% S      D x D  covariance matrix of the state distribution
% cost   .      cost structure
%   z    D x 1  target state (default: zero vector)
%   W    D x D  weight matrix (default: identity matrix)
% mu     s      expected cost
% dmudm  1 x D  derivative of expected loss wrt input mean
% dmudS  D x D  derivative of expected loss wrt input covariance matrix
% s2     s      variance of reward
% ds2dm  1 x D  derivative of variance of loss wrt input mean
% ds2dS  D x D  derivative of variance of loss wrt input covariance matrix
% c      D x 1  inv(S) times input-output covariance
% dcdm   D x D  derivative of c wrt input mean
% dcds   DxD^2  derivative of c wrt input variance
%
% Copyright (C) 2008-2014 by Carl Edward Rasmussen & Marc Deisenroth 2012-05-23

D = length(m);                                             % get state dimension
if isfield(cost,'W'); W = cost.W; else W = eye(D); end         % set defaults ..
if isfield(cost,'z'); z = cost.z; else z = zeros(D,1); end     % .. if necessary

SW = S*W;
iSpW = W/(eye(D)+SW);
mu = -exp(-(m-z)'*iSpW*(m-z)/2)/sqrt(det(eye(D)+SW));    % expected loss minus 1

if nargout > 1                                   % dervivatives of expected loss
  dmudm = -mu*(m-z)'*iSpW;                                      % wrt input mean
  dmudS = mu*(iSpW*(m-z)*(m-z)'-eye(D))*iSpW/2;    % wrt input covariance matrix
end

if nargout > 3                                                % variance of loss
  i2SpW = W/(eye(D)+2*SW);
  r2 = exp(-(m-z)'*i2SpW*(m-z))/sqrt(det(eye(D)+2*SW));
  s2 = r2 - mu^2;
  if s2 < 1e-12; s2 = 0; end                             % for numerical reasons
end

if nargout > 4                                         % derivatives of variance
  ds2dm = -2*r2*(m-z)'*i2SpW-2*mu*dmudm;                        % wrt input mean
  ds2dS = r2*(2*i2SpW*(m-z)*(m-z)'-eye(D))*i2SpW-2*mu*dmudS;             % wrt S
end

if nargout > 6                            % inv(S) times input-output covariance
  t = W*z - iSpW*(SW*z+m);
  c = mu*t;
  dcdm = t*dmudm - mu*iSpW;
  dcds = -mu*(bsxfun(@times,iSpW,permute(t,[3,2,1])) + ...
                                     bsxfun(@times,permute(iSpW,[1,3,2]),t'))/2;
  dcds = bsxfun(@times,t,dmudS(:)') + reshape(dcds,D,D^2);
end

mu = 1 + mu;                              % fix the off-set on the expected loss
