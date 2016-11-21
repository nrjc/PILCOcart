function [M, S, V] = gp3(dynmodel, m, s)

% Compute joint predictions for multiple GPs with uncertain inputs. If
% dynmodel.nigp exists, individual noise contributions are added. Predictive
% variances contain uncertainty about the function, but no noise.
%
% dynmodel  dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%     .m    (optional) D-by-1 linear weights for the GP mean
%     .b    (optional) 1-by-1 biases for the GP mean
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%   nigp    optional, n by E matrix of individual noise variance terms
%
% m         D by 1 vector, mean of the test distribution
% s         D by D covariance matrix of the test distribution
%
% M         E by 1 vector, mean of pred. distribution
% S         E by E matrix, covariance of the pred. distribution
% V         D by E inv(s) times covariance between input and prediction
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
% Andrew McHutchon, & Joe Hall 2012-07-09

persistent K iK beta oldh oldn;
[n, D] = size(dynmodel.inputs);    % number of examples and dimension of inputs
[n, E] = size(dynmodel.target);      % number of examples and number of outputs
h = dynmodel.hyp;                              % short hand for hyperparameters
if ~isfield(h,'m') && ~isfield(h,'b'); [M S V] = gp0(dynmodel,m,s); return; 
elseif ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); 
elseif ~isfield(h,'b'); [h.b] = deal(0); end

% if necessary: re-compute cashed variables
if isempty(iK) || any(unwrap(h) ~= oldh) || n ~= oldn
  oldh = unwrap(h); oldn = n; iK = zeros(n,n,E); K = zeros(n,n,E); beta = zeros(n,E);
  
  for i=1:E                                              % compute K and inv(K)
    inp = bsxfun(@times,dynmodel.inputs,exp(-h(i).l'));
    K(:,:,i) = exp(2*h(i).s-maha(inp,inp)/2);
    if isfield(dynmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n) + diag(dynmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n))';
    end
    iK(:,:,i) = L'\(L\eye(n));
    y = dynmodel.target(:,i) - dynmodel.inputs*h(i).m - h(i).b;
    beta(:,i) = L'\(L\y);
  end
end

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);
a = zeros(D,E); M1 = zeros(E,1);

inp = bsxfun(@minus,dynmodel.inputs,m');                    % centralize inputs

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  iL = diag(exp(-h(i).l));
  in = inp*iL;
  B = iL*s*iL+eye(D); 
  
  t = in/B;     % in.*t = (x-m) (S+L)^-1 (x-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*iL;
  c = exp(2*h(i).s)/sqrt(det(B));   % sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  V(:,i) = tL'*lb*c + h(i).m;            % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
  LBL = iL*(B\iL); xm = dynmodel.inputs'*lb*c; 
  a(:,i) = diag(exp(2*h(i).l))*LBL*m*M1(i) + s*LBL*xm;
end

il = exp(-2*[h.l]);inpil = bsxfun(@times,inp,permute(il,[3,1,2])); % N-by-D-by-E
for i=1:E                  % compute predictive covariance, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(il(:,i)+il(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    S(i,j) = beta(:,i)'*L*beta(:,j)*t;
    if i==j; S(i,i) = S(i,i) - t*sum(sum(iK(:,:,i).*L)); end
    S(i,j) = S(i,j) + h(i).m'*(a(:,j) - m*M1(j)) + h(j).m'*(a(:,i) - m*M1(i));
    S(j,i) = S(i,j);
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);
end

S = S - M1*M1' + [h.m]'*s*[h.m];                           % centralize moments
