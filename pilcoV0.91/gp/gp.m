function [nlml, dnlml] = gp(h, x, y, mS)

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
% mS     1x1  mean switch parameter: 1 = train GP mean; 0 = fix GP mean
% nlml   1x1  negative log marginal likelihood
% dnlml       struct (as h) with derivatives of nlml wrt entries in h
%
% (C) copyright 2013 by Carl Edward Rasmussen and Andrew McHutchon, 2013-11-07

[n, D] = size(x);                 % number n, and dimension D of training cases
if isfield(h,'m'), y = y - x*h.m; end          % if necessary, subtract mean ..
if isfield(h,'b'), y = y - h.b; end                  % .. and bias from targets
z = bsxfun(@rdivide,x,exp(h.l)');       % scale the inputs by the length scales

K = exp(2*h.s-maha(z,z)/2);                      % noise-free covariance matrix
L = chol(K + exp(2*h.n)*eye(n))';     % cholesky of the noisy covariance matrix
alpha = solve_chol(L',y);

nlml = y'*alpha/2 + sum(log(diag(L))) + n*log(2*pi)/2;       % neg log marg lik

if nargout > 1     % ---------------------------------- derivative computations
  W = L'\(L\eye(n))-alpha*alpha';                  % precompute for convenience
  if isfield(h,'m'); if mS, dnlml.m = -x'*alpha; 
      else dnlml.m = zeros(D,1); end; end
  if isfield(h,'b'); if mS, dnlml.b = -sum(alpha); 
      else dnlml.b = 0; end; end
  dnlml.l = sq_dist(z',[],K.*W)/2;
  dnlml.s = K(:)'*W(:);
  dnlml.n = exp(2*h.n)*trace(W);
end