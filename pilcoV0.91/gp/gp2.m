%% gp2.m
% *Summary:* Compute joint predictions and derivatives for multiple GPs
% with uncertain inputs. Does not consider the uncertainty about the underlying
% function (in prediction), hence, only the GP mean function is considered.
% Therefore, this representation is equivalent to a regularized RBF
% network.
% If gpmodel.nigp exists, individial noise contributions are added.
%
%
%   function [M, S, V] = gp2(gpmodel, m, s)
%
% *Input arguments:*
%
%   gpmodel    GP model struct
%   hyp(i)     struct array of GP hyper-parameters                   [ 1  x  E ]
%       .l     log lengthscales                                      [ D  x  1 ]
%       .s     log signal standard deviation                         [ 1  x  1 ]
%       .n     log noise standard deviation                          [ 1  x  1 ]
%       .m     linear weights for the GP mean                        [ D  x  1 ]
%       .b     biases for the GP mean                                [ 1  x  1 ]
%     inputs   training inputs                                       [ n  x  D ]
%     targets  training targets                                      [ n  x  E ]
%   m          mean of the test distribution                         [ D  x  1 ]
%   s          covariance matrix of the test distribution            [ D  x  D ]
%
% *Output arguments:*
%
%   M          mean of pred. distribution                            [ E  x  1 ]
%   S          covariance of the pred. distribution                  [ E  x  E ]
%   V          inv(s) times covariance between input and output      [ D  x  E ]
%
%
% Copyright (C) 2008-2014 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2043-03-03
%
%% High-Level Steps
% # Compute predicted mean and inv(s) times input-output covariance
% # Compute predictive covariance matrix, non-central moments
% # Centralize moments

function [M, S, V] = gp2(gpmodel, x, m, s)
%% Code
[n, D, pE] = size(x); E = size(gpmodel.beta,2);
h = gpmodel.hyp; beta = gpmodel.beta;
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end
if ~isfield(h,'b'); [h.b] = deal(0); end

M = zeros(E,1); V = zeros(D,E); S = zeros(E);
k = zeros(n,E); a = zeros(D,E); M1 = zeros(E,1);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E


% 2) Compute predicted mean and inv(s) times input-output covariance
for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l));          % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;       % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D);               % Lambda^-1/2 * S * *Lambda^-1/2 + I
  
  t = in/B;                         % in.*t = (X-m) (S+L)^-1 (X-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il;
  c = exp(2*h(i).s)/sqrt(det(B));   % = sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  V(:,i) = tL'*lb*c + h(i).m;            % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
  liBl = il*(B\il); xm = x(:,:,min(i,pE))'*lb*c; 
  a(:,i) = diag(exp(2*h(i).l))*liBl*m*M1(i) + s*liBl*xm;
end

% 3) Compute predictive covariance, non-central moments
iL = exp(-2*[h.l]);inpiL = bsxfun(@times,inp,permute(iL,[3,1,2])); % N-by-D-by-E
for i=1:E                  % compute predictive covariance, non-central moments
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); t = 1/sqrt(det(R));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(inpiL(:,:,i),-inpiL(:,:,j),R\s/2));
    S(i,j) = beta(:,i)'*L*beta(:,j)*t;                  % variance of the mean
    S(i,j) = S(i,j) + h(i).m'*(a(:,j) - m*M1(j)) + h(j).m'*(a(:,i) - m*M1(i));
    S(j,i) = S(i,j);
  end
  
  S(i,i) = S(i,i) + 1e-6;          % add small jitter for numerical reasons
  
end

% 4) Centralize moments
S = S - M1*M1' + [h.m]'*s*[h.m];                           % centralize moments
