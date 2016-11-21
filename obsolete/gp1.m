function [M, S, V] = gp1(gpmodel, m, s)

% Compute joint predictions for the FITC sparse approximation to multiple GPs
% with uncertain inputs. If dynmodel.nigp exists, individual noise contribu-
% tions are added. 
%
% dynmodel  dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   induce  inducing inputs                                         [ np x  D ]
%   nigp    (optional) individual noise variance terms              [ n  x  E ]
% m         mean of the test distribution                           [ D       ]
% s         covariance matrix of the test distribution              [ D  x  D ]
%
% M         mean of pred. distribution                              [ E       ]
% S         covariance of the pred. distribution                    [ E  x  E ]
% V         inv(s) times covariance between input and output        [ D  x  E ]
%
% Copyright (C) 2008-2012 by Marc Deisenroth, Carl Edward Rasmussen,
% Andrew McHutchon, & Joe Hall  2012-07-09

if ~isfield(gpmodel,'induce') || numel(gpmodel.induce)==0, 
    [M, S, V] = gp0(gpmodel, m, s); return; end

persistent iK2 beta oldh;
ridge = 1e-6;                        % jitter to make matrix better conditioned
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
E = size(gpmodel.target,2);         % number of examples and number of outputs
h = gpmodel.hyp; input = gpmodel.inputs; target = gpmodel.target;

[np pD pE] = size(gpmodel.induce);     % number of pseudo inputs per dimension
pinput = gpmodel.induce;                                   % all pseudo inputs

if numel(unwrap(h)) ~= numel(oldh) || isempty(iK2) || ... % if necessary
                    any(unwrap(h) ~= oldh) || numel(iK2) ~=E*np^2
  oldh = unwrap(h);                                    % compute K, inv(K), inv(K2)
  iK = zeros(np,n,E); iK2 = zeros(np,np,E); beta = zeros(np,E);
    
  for i=1:E
    pinp = bsxfun(@times,pinput(:,:,min(i,pE)),exp(-h(i).l'));
    inp = bsxfun(@times,input,exp(-h(i).l'));
    Kmm = exp(2*h(i).s-maha(pinp,pinp)/2) + ridge*eye(np);  % add small ridge
    Kmn = exp(2*h(i).s-maha(pinp,inp)/2);
    L = chol(Kmm)';
    V = L\Kmn;                                             % inv(sqrt(Kmm))*Kmn
    G = exp(2*h(i).s)-sum(V.^2);
    if isfield(gpmodel,'nigp'); G = G + gpmodel.nigp(:,i)'; end
    G = sqrt(1+G/exp(2*h(i).n));
    V = bsxfun(@rdivide,V,G);
    Am = chol(exp(2*h(i).n)*eye(np) + V*V')';
    At = L*Am;                                    % chol(sig*B) [thesis, p. 40]
    iAt = At\eye(np);
% The following is not an inverse matrix, but we'll treat it as such: multiply
% the targets from right and the cross-covariances left to get predictive mean.
    iK(:,:,i) = ((Am\(bsxfun(@rdivide,V,G)))'*iAt)';
    beta(:,i) = iK(:,:,i)*target(:,i);      
    iB = iAt'*iAt.*exp(2*h(i).n);              % inv(B), [Ed's thesis, p. 40]
    iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
  end
end

k = zeros(np,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E); 
inp = zeros(np,D,E);

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  inp(:,:,i) = bsxfun(@minus,pinput(:,:,min(i,pE)),m');
 
  iL = diag(exp(-h(i).l));
  in = inp(:,:,i)*iL;
  B = iL*s*iL+eye(D); 
  
  t = in/B;
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*iL;
  c = exp(2*h(i).s)/sqrt(det(B));
  
  M(i) = sum(lb)*c;                                            % predicted mean
  V(:,i) = tL'*lb*c;                     % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
end

il = exp(-2*[h.l]);inpil = bsxfun(@times,inp,permute(il,[3,1,2])); % N-by-D-by-E
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(il(:,i)+il(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    S(i,j) = beta(:,i)'*L*beta(:,j)*t; S(j,i) = S(i,j);
    if i==j; S(i,i) = S(i,i) - t*sum(sum(iK2(:,:,i).*L)); end
  end

  S(i,i) = S(i,i) + exp(2*h(i).s);
end

S = S - M*M';                                               % centralize moments
