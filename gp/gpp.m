function [M, S, C] = gpp(gp, m, s)
% Compute joint predictions for multiple GPs with uncertain inputs. Predictive
% variances contain uncertainty about the function, but no noise.
%
% gp               Gaussian process model struct
%   hyp     1 x E  struct array of GP hyper-parameters
%     l     D x 1  log lengthscales
%     s     1 x 1  log signal standard deviation
%     n     1 x 1  log noise standard deviation
%   inputs  n x D  matrix of training inputs
%   target  n x E  matrix of training targets
%   W       nxnxE  inverse covariance matrix
%   beta    n x E  iK*(targets - mean function of inputs)
% m         D x 1  mean of the test distribution
% s         D x D  covariance matrix of the test distribution
% M         E x 1  mean of pred. distribution
% S         E x E  covariance of the pred. distribution
% C         D x E  inv(s) times covariance between input and output
%
% Copyright (C) 2015, Carl Edward Rasmussen, Rowan McAllister 2015-07-10

if numel(gp.induce) > 0; x = gp.induce; else x = gp.inputs; end

[n, ~, pE] = size(x); D = size(x,2); E = size(gp.beta,2);
h = gp.hyp; iK = gp.W; beta = gp.beta;
M = zeros(E,1); C = zeros(D,E); S = zeros(E); k = zeros(n,E);

inp = bsxfun(@minus,x,m');     % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l));                                        % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;                             % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D);                      % Lambda^-1/2 * V * *Lambda^-1/2 + I
  t = in/B;                                      % in.*t = (X-m) (V+L)^-1 (X-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il;
  c = exp(2*h(i).s)/sqrt(det(B));                   % = sf2/sqrt(det(V*iL + I))
  M(i) = sum(lb)*c;                                            % predicted mean
  C(:,i) = tL'*lb*c;                     % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
end

hl = [h.l];
iL = exp(-2*hl); xiL = bsxfun(@times,inp,permute(iL,[3,1,2]));
for i=1:E                  % compute predictive covariance, non-central moments
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); t = 1/sqrt(det(R));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)') + maha(xiL(:,:,i),-xiL(:,:,j),R\s/2));
    S(i,j) = beta(:,i)'*L*beta(:,j)*t;                   % variance of the mean
    S(j,i) = S(i,j);
  end
  S(i,i) = S(i,i) + exp(2*h(i).s) - t*sum(sum(iK(:,:,i).*L)); % last L has i==j
end

S = S - M*M';                                              % centralize moments
M = M + [h.m]'*m + [h.b]';                           % add linear contributions
S = S + C'*s*[h.m] + [h.m]'*s*C + [h.m]'*s*[h.m];
C = C + [h.m];