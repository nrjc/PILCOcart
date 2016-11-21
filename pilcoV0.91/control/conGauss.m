function [M, S, C] = conGauss(policy, m, s)
% Compute joint predictions for multiple Gaussian Radial Basis Functions (RBFs)
% with uncertain inputs.
%
% policy        policy struct
%   .p          parameters to optimise
%     .cen      (either) n by D matrix of input locations for basis functions
%     .ll       (either) D by E matrix of log lengthscales
%     .w        n by E matrix of basis functions weights
%   .cen        (or) if cen appears here it will not be optimised
%   .ll         (or) if ll appears here it will not be optimised
%
% m             D by 1 vector, mean of the test distribution
% s             D by D covariance matrix of the test distribution
%
% M             E by 1 vector, mean of pred. distribution
% S             E by E matrix, covariance of the pred. distribution
% V             D by E inv(s) times covariance between input and prediction
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
% Joe Hall, and Andrew McHutchon. 2012-07-23

[n, E] = size(policy.p.w);          % number of bases and number of outputs
if isfield(policy,'cen'); cen = policy.cen; else cen = policy.p.cen; end
if isfield(policy,'ll'); ll = policy.ll; else ll = policy.p.ll; end
w = policy.p.w;                     % weights on individual basis functions
D = size(ll,1);            % dimension of inputs


% initialisations
M = zeros(E,1); S = zeros(E); C = zeros(D,E); k = zeros(n,E,D);

inp = bsxfun(@minus,cen,m');                    % centralize inputs

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  LL = diag(exp(-ll(:,i)));
  in = inp*LL;
  B = LL*s*LL+eye(D); 
  tt = in/B;
  l = exp(-sum(in.*tt,2)/2); lb = l.*w(:,i);
  tL = tt*LL;
  c = 1/sqrt(det(B));
  
  M(i) = sum(lb)*c;                                            % predicted mean
  C(:,i) = tL'*lb*c;                    % inv(s) times input-output covariance
  k(:,i) = -sum(in.*in,2)/2;
end

for i=1:E                  % compute predictive covariance, non-central moments
  ii = bsxfun(@rdivide,inp,exp(2*ll(1:D,i)'));
  
  for j=1:i
    R = s*diag(exp(-2*ll(1:D,i))+exp(-2*ll(1:D,j)))+eye(D); 
    t = 1/sqrt(det(R));
    ij = bsxfun(@rdivide,inp,exp(2*ll(1:D,j)'));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    A = w(:,i)*w(:,j)';     A = A.*L;
    S(i,j) = t*sum(sum(A)); S(j,i) = S(i,j);   
  end
end

S = S - M*M' + 1e-6*eye(E);               % centralize moments...and add jitter
