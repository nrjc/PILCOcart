function x = gaussian(m, S, n);

if nargin < 3, n = 1; end

x = bsxfun(@plus, m(:), chol(S)'*randn(size(S,2),n));
