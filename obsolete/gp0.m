function [M, S, V] = gp0(gpmodel, m, s)

% Compute joint predictions for multiple GPs with uncertain inputs. If
% dynmodel.nigp exists, individual noise contributions are added. Predictive
% variances contain uncertainty about the function, but no noise.
%
% dynmodel  dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   nigp    (optional) individual noise variance terms              [ n  x  E ]
% m         mean of the test distribution                           [ D       ]
% s         covariance matrix of the test distribution              [ D  x  D ]
%
% M         mean of pred. distribution                              [ E       ]
% S         covariance of the pred. distribution                    [ E  x  E ]
% V         inv(s) times covariance between input and output        [ D  x  E ]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
% Andrew McHutchon, & Joe Hall 2013-07-05

persistent K iK beta oldh oldn;
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
[n, E] = size(gpmodel.target);      % number of examples and number of outputs
h = gpmodel.hyp;                              % short hand for hyperparameters

% if necessary: re-compute cashed variables
if isempty(iK) || any(unwrap(h) ~= oldh) || n ~= oldn
  oldh = unwrap(h); oldn = n;                                               
  iK = zeros(n,n,E); K = zeros(n,n,E); beta = zeros(n,E);
  
  for i=1:E                                              % compute K and inv(K)
    inp = bsxfun(@times,gpmodel.inputs,exp(-h(i).l'));
    K(:,:,i) = exp(2*h(i).s-maha(inp,inp)/2);
    if isfield(gpmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n) + diag(gpmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n))';
    end
    iK(:,:,i) = L'\(L\eye(n));
    beta(:,i) = L'\(L\gpmodel.target(:,i));
  end
end

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);

inp = bsxfun(@minus,gpmodel.inputs,m');                     % centralize inputs

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  iL = diag(exp(-h(i).l)); % inverse length-scales
  in = inp*iL;
  B = iL*s*iL+eye(D); 
  
  t = in/B;
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tiL = t*iL;
  c = exp(2*h(i).s)/sqrt(det(B));
  
  M(i) = sum(lb)*c;                                            % predicted mean
  V(:,i) = tiL'*lb*c;                     % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
end

il = exp(-2*[h.l]);inpil = bsxfun(@times,inp,permute(il,[3,1,2])); % N-by-D-by-E
for i=1:E                  % compute predictive covariance, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(il(:,i)+il(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    S(i,j) = beta(:,i)'*L*beta(:,j)*t; S(j,i) = S(i,j);
    if i==j; S(i,i) = S(i,i) - t*sum(sum(iK(:,:,i).*L)); end
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);
end

S = S - M*M';                                              % centralize moments