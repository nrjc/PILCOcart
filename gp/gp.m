function [nlml, dnlml] = gp(h, x, y, fixLin)
% gp - compute the negative log marginal likelihood and its derivatives wrt
% hyperparemeters. The (optional) mean function is linear in the inputs and the
% covariance function is squared exponential with ARD lengthscale parameters.
%
% h              hyperparameter struct
%   m     D x 1  [optional] mean function coefficients
%   b     1 x 1  [optional] mean function bias
%   l     D x 1  ARD log lengthscale parameters
%   s     1 x 1  log of signal std dev 
%   n     1 x 1  log of noise std dev
% x       n x D  training inputs
% y       n x 1  training targets
% fixLin  1 x 1  switch: 0 or [] to train the linear weights, 1 to fix
% nlml    1 x 1  negative log marginal likelihood
% dnlml          struct (as h) with derivatives of nlml wrt entries in h
%
% Copyright (C) 2013-2016 Carl Edward Rasmussen & Andrew McHutchon, 2016-03-24

[n, D] = size(x);                 % number n, and dimension D of training cases
if isfield(h,'m'), y = y - x*h.m; end          % if necessary, subtract mean ..
if isfield(h,'b'), y = y - h.b; end                  % .. and bias from targets
z = bsxfun(@times,x,exp(-h.l)');        % scale the inputs by the length scales
K = exp(2*h.s-maha(z,z)/2);                      % noise-free covariance matrix
L = chol(K + exp(2*h.n)*eye(n))';     % cholesky of the noisy covariance matrix
alpha = solve_chol(L',y);
nlml = y'*alpha/2 + sum(log(diag(L))) + n*log(2*pi)/2;       % neg log marg lik

if nargout > 1                                        % derivative computations
  W = L'\(L\eye(n))-alpha*alpha'; KW = K.*W;       % precompute for convenience
  if isfield(h,'m'); 
    if fixLin, dnlml.m = 0*h.m; else dnlml.m = -x'*alpha; end
  end
  if isfield(h,'b'),
    if fixLin, dnlml.b = 0; else dnlml.b = -sum(alpha); end
  end
  dnlml.l = sum((bsxfun(@times, sum(KW,2), z)-KW*z).*z,1)';
  dnlml.s = sum(sum(KW));
  dnlml.n = exp(2*h.n)*trace(W);
end
