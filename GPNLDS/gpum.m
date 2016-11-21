function [nlml, dnlml] = gpum(h, x, y)

% gp - compute the negative log marginal likelihood and its derivatives wrt
% hyperparemeters. The (optional) mean function is linear in the inputs and the
% covariance function is squared exponential with ARD lengthscale parameters.
%
% h           struct of hyperparameters, with fields
%   m    Dx1  [optional] mean function coefficients
%   b    1x1  [optional] mean function bias
%   l    Dx1  ARD log lengthscale parameters
%   s    1x1  log of signal std dev 
%   n    1x1  log of noise std dev
% x      nxD  training inputs
% y      nx1  training targets
% nlml   1x1  negative log marginal likelihood
% dnlml       struct (as h) with derivatives of nlml wrt entries in h
%
% (C) copyright 2013 by Carl Edward Rasmussen, 2013-07-05

global usemean

[n, D] = size(x);                 % number n, and dimension D of training cases
if isfield(h,'m'), y = y - x*h.m; end          % if necessary, subtract mean ..
if isfield(h,'b'), y = y - h.b; end                  % .. and bias from targets
z = bsxfun(@rdivide,x,exp(h.l)');       % scale the inputs by the length scales

K = exp(2*h.s-maha(z,z)/2);                      % noise-free covariance matrix
L = chol(K + exp(2*h.n)*eye(n))';     % cholesky of the noisy covariance matrix
alpha = solve_chol(L',y);

nlml = y'*alpha/2 + sum(log(diag(L))) + n*log(2*pi)/2;       % neg log marg lik
if nargout > 1
  W = L'\(L\eye(n))-alpha*alpha';                  % precompute for convenience
  if isfield(h,'m'), dnlml.m = -x'*alpha; end  
  if isfield(h,'b'), dnlml.b = -sum(alpha); end
  dnlml.l = sq_dist(z',[],K.*W)/2;
  dnlml.s = K(:)'*W(:);
  dnlml.n = exp(2*h.n)*trace(W);
  if 2>usemean; dnlml.m = 0*dnlml.m; dnlml.b = 0*dnlml.b; end
end
