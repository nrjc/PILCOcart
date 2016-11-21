% A      = nigp(hyp, inputs, target, A);
% [f df] = nigp(hyp, inputs, target, A);
% [mu v] = nigp(hyp, inputs, target, A, test);
%
% Copyright (C) 2015 Carl Edward Rasmussen, 2015-07-16

function [varargout] = nigp(hyp, inputs, target, A, test);

[N, D] = size(inputs); E = length(hyp);
if nargin < 4, A = zeros(N,E,E); for i=1:E, A(:,i,i) = 1; end; end   % default A
n2 = exp(2*[hyp(:).n]');
if nargout == 1
  for e = 1:E
    y = target(:,e) - inputs*hyp(e).m - hyp(e).b;
    l = exp(hyp(e).l); s2 = exp(2*hyp(e).s);
    x = bsxfun(@rdivide, inputs, l');  
    K = s2*exp(-maha(x,x)/2) + diag(A(:,:,e)*n2);
    for ee = 1:E
      d = I(ee);
      dK = bsxfun(@minus,x(:,d),x(:,d)').*exp(-maha(x,x)/2);
      B = dK/K;
      By = B*y*s2/l(d) + hyp(e).m(d);
      A(:,ee,e) = (ee==e) + By.^2 + (s2 - s2^2*sum(B.*dK,2))/l(d)^2;
    end
  end
  varargout(1) = {A};
elseif nargin < 5
  dnlml = hyp; for e = 1:E, dnlml(e).n = 0; end
  for e = 1:E
    y = target(:,e) - inputs*hyp(e).m - hyp(e).b;
    l = exp(hyp(e).l); s2 = exp(2*hyp(e).s);
    x = bsxfun(@rdivide, inputs, l');  
    K = s2*exp(-maha(x,x)/2);
    L = chol(K + diag(A(:,:,e)*n2))';  % cholesky of the noisy covariance matrix
    alpha = solve_chol(L', y);
    nlml(e) = y'*alpha/2 + sum(log(diag(L))) + N*log(2*pi)/2; % neg log marg lik
    W = L'\(L\eye(N))-alpha*alpha';                 % precompute for convenience
    dnlml(e).m = -inputs'*alpha;
    dnlml(e).b = -sum(alpha);
    dnlml(e).l = sq_dist(x',[],K.*W)/2;
    dnlml(e).s = K(:)'*W(:);
    t = diag(W)'*A(:,:,e).*n2';
    for d = 1:D
      dnlml(d).n = dnlml(d).n + t(d); 
    end
  end
  varargout(1) = {sum(nlml)}; varargout(2) = {dnlml};
else
  for e = 1:E
    y = target(:,e) - inputs*hyp(e).m - hyp(e).b;
    l = exp(hyp(e).l); s2 = exp(2*hyp(e).s);
    z = bsxfun(@rdivide, test, l');  
    x = bsxfun(@rdivide, inputs, l');  
    K = s2*exp(-maha(x,x)/2) + diag(A(:,:,e)*n2);
    Ks = s2*exp(-maha(z,x)/2);
    mu(:,e) = Ks*(K\y);
    v(:,e) = s2 - sum(Ks/K.*Ks,2);
  end
  varargout(1) = {mu}; varargout(2) = {v};
end

function K = c(hyp, angi, inputs, test)
[N, D] = size(inputs); K = zeros(N, N);
for d = 1:D
  if ismember(d, angi)
    z = inputs(:,d)/2;
    K = K + (2*sin(bsxfun(@minus, z, z'))/hyp.l(d)).^2;
  else
    z = inputs(:,d)/hyp.l(d);
    K = K + bsxfun(@minus, z, z').^2;
  end
end
K = exp(2*hyp.s-K/2);