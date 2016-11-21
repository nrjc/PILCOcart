function [M, S, C, V] = gph(gpmodel, m, s, v, combineSV, approxSV)

% Compute joint predictions for multiple GPs with hierarchical uncertain inputs.
%
% gpmodel          dynamics model struct
%   hyp     1 x E  struct array of GP hyper-parameters
%     l     D x 1  log lengthscales
%     s            log signal standard deviation
%     n            log noise standard deviation
%     m     D x 1  linear weights for GP mean
%     b            bias for GP mean
%   inputs  n x D  or
%           nxDxE  training inputs, possibly separate per target
%   target  n x E  training targets
%   W       nxnxE  inverse noisy covariance matrices
%   beta    n x E  W*(targets - mean function of inputs)
% m         D x 1  mean     of         test input if nargin == 3, or
%           D x 1  mean     of mean of test input if nargin == 4
% s         D x D  variance of         test input if nargin == 3, or
%           D x D  variance of mean of test input if nargin == 4
% v         D x D  variance of         test input if nargin == 4
% combineSV bool   output S (not V) where S is instead S+V
% approxSV  bool   compute an inexpensive-and-approximate S and V
% M         E x 1  mean     or mean of mean of prediction
% S         E x E  variance or variance of mean of prediction
% C         D x E  inv(s) times mean input - mean output cov, equivalently
%                  inv(v) times expected[input-output covariance]
% V         E x E  mean of variance of prediction
%
% See also <a href="gph.pdf">gph.pdf</a>, GPHD.M, GPHT.M.
% Copyright (C) 2014 by Carl Edward Rasmussen and Rowan McAllister 2016-01-20

if isprop(gpmodel,'induce') && numel(gpmodel.induce) > 0
  x = bsxfun(@minus, gpmodel.induce, m');            % x is either nxD or nxDxE
else
  x = bsxfun(@minus, gpmodel.inputs, m');                            % x is nxD
end

[n, D, pE] = size(x); E = size(gpmodel.beta,2); C = zeros(D,E);
if nargin < 4; v = zeros(D); end
if nargin < 5; combineSV = false; end
if nargin < 6; approxSV = false; end
h = gpmodel.hyp; iK = gpmodel.W; beta = gpmodel.beta;
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end
if ~isfield(h,'b'); [h.b] = deal(0); end
ss2 = exp(2*bsxfun(@plus,[h.s]',[h.s]));

[lq,c] = q(x, [h.l], s+v);                      % first, compute predicted mean
qb = exp(lq).*beta;
M = (exp(c+2*[h.s]).*sum(qb,1))';              % mean without mean contribution

for i=1:E                           % then inv(s) times input-output covariance
  il=diag(exp(-2*h(i).l)); il=il/(eye(D)+(s+v)*il);
  C(:,i) = il*(x(:,:,min(i,pE))'*qb(:,i));
end
C = bsxfun(@times, C, exp(c+2*[h.s]));   % covariance without mean contribution

if approxSV                                              % variance of the mean
  S = (C+[h.m])'*s*(C+[h.m]);
else
  bQb = Q(x, [h.l], v, s, iK, beta, ~approxSV);
  S = ss2.*bQb - M*M' + C'*s*[h.m] + [h.m]'*s*C + [h.m]'*s*[h.m];
end

if approxSV                                     % finally, mean of the variance
  [~,tiKQ] = Q(x, [h.l], zeros(D), s+v, iK, beta, ~approxSV);
  V = (C+[h.m])'*v*(C+[h.m]) + ss2.*diag(exp(-2*[h.s])-tiKQ);
else
  V = bQb;                                        
  [bQb,tiKQ] = Q(x, [h.l], zeros(D), s+v, iK, beta, ~approxSV);
  V = ss2.*(bQb - V + diag(exp(-2*[h.s])-tiKQ)) + C'*v*[h.m] + [h.m]'*v*C + [h.m]'*v*[h.m];
end

M = M + [h.m]'*m + [h.b]';                  % add contribution of mean function
C = C + [h.m];                        % add the mean contribution to covariance
if nargout < 4 || combineSV; S = S + V; V = nan(E); end


% The 'q' function: return z (n x E) of exp negative quatratics and c (1 x E)
% of -log(det)/2 separately.
function [z,c] = q(x, L, V)
[n, D, pE] = size(x); E = size(L,2); z = zeros(n,E); c = zeros(1,E);
for i=1:E
  il = diag(exp(-L(:,i)));                                        % Lambda^-1/2
  in = x(:,:,min(i,pE))*il;                               % (X - m)*Lambda^-1/2
  B = il*V*il+eye(D);                       % Lambda^-1/2 * V * Lambda^-1/2 + I
  z(:,i) = -sum(in.*(in/B),2)/2;
  c(i) = -sum(log(diag(chol(B))));                  % -log(det(Lambda\V + I))/2
end


% The 'Q' function: return all quadratics of beta with Q, bQb (E x E) and the
% traces of the products of iK and Q, tikQ (1 x E).
function [bQb,tiKQ] = Q(x, L, V, s, iK, beta, computebQb)
[n, D, pE] = size(x); E = size(L,2);
bQb = nan(E); tiKQ = nan(1,E); iL = zeros(D,D,E); xiL = zeros(n,D,E);
[lq,c] = q(x, L, V);
for i=1:E
  il = diag(exp(-2*L(:,i)));
  iL(:,:,i) = il/(V*il + eye(D));
  xiL(:,:,i) = x(:,:,min(pE,i))*iL(:,:,i);
end
for i=1:E
  for j=1:i
    if ~computebQb && j ~= i; continue; end
    R = s*(iL(:,:,i)+iL(:,:,j))+eye(D); t = exp(c(i)+c(j))/sqrt(det(R));
    Q = exp(bsxfun(@plus,lq(:,i),lq(:,j)')+maha(xiL(:,:,i),-xiL(:,:,j),R\s/2));
    if computebQb
      bQb(i,j) = beta(:,i)'*Q*beta(:,j)*t; bQb(j,i) = bQb(i,j);
    end
  end
  if nargout>1, tiKQ(i) = sum(sum(iK(:,:,i).*Q))*t; end
end