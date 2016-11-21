function [M, S, V] = gpa2(gpmodel, m, s)

% Compute joint predictions for multiple GPs with first order additive squared
% exponential kernels and uncertain inputs. Predictive variances contain
% neither uncertainty about the underlying function nor measurement noise.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters ( P = 2*D+1 )                      [ P  x  E ]
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   noise   (optional) individual noise variance terms              [ n  x  E ]
% m         mean of the test distribution                           [ D       ]
% s         covariance matrix of the test distribution              [ D  x  D ]
%
% M         mean of pred. distribution                              [ E       ]
% S         covariance of the pred. distribution                    [ E  x  E ]
% V         inv(s) times covariance between input and output        [ D  x  E ]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen & Marc Deisenroth 2012-01-81
% Edited by Joe Hall 2012-03-21

persistent K iK beta oldX oldn;
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
[n, E] = size(gpmodel.target);      % number of examples and number of outputs
D2 = 2*D; X = reshape(gpmodel.hyp, D2+1, E);  % short hand for hyperparameters

% if necessary: re-compute cashed variables
if numel(X) ~= numel(oldX) || isempty(iK) || sum(any(X ~= oldX)) || n ~= oldn
  oldX = X; oldn = n;                                               
  iK = zeros(n,n,E); K = zeros(n,n,E); beta = zeros(n,E);
  
  for i=1:E                                              % compute K and inv(K)
    inp = bsxfun(@rdivide,gpmodel.inputs,exp(X(1:D,i)'));
    for d = 1:D,
      K(:,:,i) = K(:,:,i) + exp(2*X(D+d,i)-maha(inp(:,d),inp(:,d))/2);
    end
    L = chol(K(:,:,i) + exp(2*X(D2+1,i))*eye(n))';
    iK(:,:,i) = L'\(L\eye(n));
    beta(:,i) = L'\(L\gpmodel.target(:,i));
  end
end

M = zeros(E,1); V = zeros(D,E); S = zeros(E); k = zeros(n,E,D);

inp = bsxfun(@minus,gpmodel.inputs,m');                    % centralize inputs

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  for d=1:D
    L = exp(-X(d,i));
    in = inp(:,d)*L;
    B = L*s(d,d)*L + 1;
    t = in/B;
    l = exp( 2*X(D+d,i) - in.*t/2)/sqrt(B);
    lb = l.*beta(:,i); tL = t*L;
  
    M(i) = M(i) + sum(lb);                                     % predicted mean
    V(d,i) = tL'*lb;                     % inv(s) times input-output covariance
    k(:,i,d) = 2*X(D+d,i) - in.*in/2;
  end
end

for i=1:E                  % compute predictive covariance, non-central moments
  ii = bsxfun(@rdivide,inp,exp(2*X(1:D,i)'));
  
  for j=1:i
    ij = bsxfun(@rdivide,inp,exp(2*X(1:D,j)'));    
    BB = beta(:,i)*beta(:,j)';

    for d=1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end
      for e=1:P
        if d==e
          sde = s(d,d); p = 1; pp = 1;
          ii_ = ii(:,d); ij_ = ij(:,e);
          eXi = exp(-2*X(d,i)); eXj = exp(-2*X(e,j));
        else
          sde = s([d e],[d e]); p = 2;
          ii_ = [ii(:,d) zeros(n,1)]; ij_ = [zeros(n,1) ij(:,e)];
          eXi = [exp(-2*X(d,i)) 0]; eXj = [0 exp(-2*X(e,j))];
        end
        R = sde*diag(eXi + eXj) + eye(p);
        t = 1/sqrt(det(R));
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') + maha(ii_,-ij_,R\sde/2));
        ssA = sum(sum(BB.*L));
        S(i,j) = S(i,j) + pp*t*ssA; S(j,i) = S(i,j);     % predicted covariance
      end % e
    end % d 
    
  end % j
end % i

S = S - M*M' + 1e-6*eye(E);               % centralize moments...and add jitter