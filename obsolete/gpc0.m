function [M, S, V] = gpc0(gpmodel, m, s)

% Compute joint predictions for multiple GPs with a combination of first order
% additive SE kernels plus a standard SE kernel and uncertain inputs. If 
% dynmodel.nigp exists, individual noise contributions are added. Predictive 
% variances contain uncertainty about the function, but no measurement noise.
%
% dynmodel  dynamics model struct
%   hyp     3D+2 by E matrix of log-hyper-parameters
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%   nigp   optional, n by E matrix of individual noise variance terms
%
% m         D by 1 vector, mean of the test distribution
% s         D by D covariance matrix of the test distribution
%
% M         E by 1 vector, mean of pred. distribution
% S         E by E matrix, covariance of the pred. distribution
% V         D by E inv(s) times covariance between input and prediction
%
% Copyright (C) 2008-2011 by Carl Edward Rasmussen & Marc Deisenroth 2012-01-11
% Edited by Joe Hall 2012-04-04

persistent K iK beta oldX oldn;
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
[n, E] = size(gpmodel.target);      % number of examples and number of outputs
D2 = 2*D; D3 = 3*D;                                              % useful terms
X = reshape(gpmodel.hyp, D3+2, E);            % short hand for hyperparameters
X1 = X([1:D D2+1:D3],:);                % hyperparameters for 1st order kernels
X2 = X([D+1:D2 D3+1],:);                 % hyperparameters for Dth order kernel

% if necessary: re-compute cashed variables
if numel(X) ~= numel(oldX) || isempty(iK) || sum(any(X ~= oldX)) || n ~= oldn
  oldX = X; oldn = n;                                               
  iK = zeros(n,n,E); K = zeros(n,n,E); beta = zeros(n,E);
  
  for i=1:E                                              % compute K and inv(K)
    inp1 = bsxfun(@rdivide,gpmodel.inputs,exp(X1(1:D,i)'));
    inp2 = bsxfun(@rdivide,gpmodel.inputs,exp(X2(1:D,i)'));
    for d = 1:D,
      K(:,:,i) = K(:,:,i) + exp(2*X1(D+d,i)-maha(inp1(:,d),inp1(:,d))/2);
    end
    K(:,:,i) = K(:,:,i) + exp(2*X2(D+1,i)-maha(inp2,inp2)/2);
    if isfield(gpmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*X(D3+2,i))*eye(n)+diag(gpmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*X(D3+2,i))*eye(n))';
    end
    iK(:,:,i) = L'\(L\eye(n));
    beta(:,i) = L'\(L\gpmodel.target(:,i));
  end
end

k = zeros(n,E,D+1); M = zeros(E,1); V = zeros(D,E); S = zeros(E);

inp = bsxfun(@minus,gpmodel.inputs,m');                    % centralize inputs

% 1) Predicted Mean and Input-Output Covariance *******************************
for i=1:E
    
  % 1a) First Order Additive Kernels ------------------------------------------
  for d=1:D
    L = exp(-X1(d,i));
    in = inp(:,d)*L;
    B = L*s(d,d)*L + 1;
    t = in/B;
    l = exp( 2*X1(D+d,i) - in.*t/2)/sqrt(B);
    lb = l.*beta(:,i); tL = t*L;
  
    M(i) = M(i) + sum(lb);                                     % predicted mean
    V(d,i) = tL'*lb;                     % inv(s) times input-output covariance
    k(:,i,d) = 2*X1(D+d,i) - in.*in/2;
  end
  
  % 1b) Full Squared Exponential Kernel ---------------------------------------
  R = s+diag(exp(2*X2(1:D,i)));
  iLam = diag(exp(-2*X2(1:D,i)));
  iR = iLam*(eye(D) - (eye(D)+s*iLam)\(s*iLam));
  iR = (iR+iR')/2;                                            % Kailath inverse
  t = inp*iR;
  l = exp(-sum(t.*inp,2)/2); lb = l.*beta(:,i);
  c = exp(2*X2(D+1,i))/sqrt(det(R))*exp(sum(X2(1:D,i)));

  M(i) = M(i) + sum(lb)*c;                                     % predicted mean
  V(:,i) = V(:,i) + t'*lb*c;             % inv(s) times input output covariance
  
  v = bsxfun(@rdivide,inp,exp(X2(1:D,i)'));
  k(:,i,D+1) = 2*X2(D+1,i)-sum(v.*v,2)/2;
end

% 2) Predictive Covariance Matrix *********************************************
for i=1:E
  ii1 = bsxfun(@rdivide,inp,exp(2*X1(1:D,i)'));
  ii2 = bsxfun(@rdivide,inp,exp(2*X2(1:D,i)'));
  
  for j=1:i
    ij1 = bsxfun(@rdivide,inp,exp(2*X1(1:D,j)'));
    ij2 = bsxfun(@rdivide,inp,exp(2*X2(1:D,j)'));
    BB = beta(:,i)*beta(:,j)';
    if i==j; BB = BB - iK(:,:,i); end           % incorporate model uncertainty
    LL = zeros(n);

    % 2a) Combo of First Order Additive Kernels -------------------------------
    for d=1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end     % fill in cross terms
      for e=1:P
        if d==e % ----------------------------------------- k_id and k_je (d==e)
          sde = s(d,d); p = 1; pp = 1;
          ii_ = ii1(:,d); ij_ = ij1(:,e);
          eXij = exp(-2*X1(d,i)) + exp(-2*X1(e,j));
        else % -------------------------------------------- k_id and k_je (d~=e)
          sde = s([d e],[d e]); p = 2;
          ii_ = [ii1(:,d) zeros(n,1)]; ij_ = [zeros(n,1) ij1(:,e)];
          eXij = [exp(-2*X1(d,i)) exp(-2*X1(e,j))];
        end
        R = sde*diag(eXij) + eye(p);
        t = 1/sqrt(det(R));
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') + maha(ii_,-ij_,R\sde/2));
        LL = LL + pp*t*L;
      end % e
    end % d
    
    % 2b) Combo of First Order and Full Squared Exponential Kernel ------------
    for d=1:D+1
      if i==j, P = 1; pp = 2; else P = 2; pp = 1; end     % fill in cross terms
      if d==D+1, P = 1; pp = 1; end
      for e=1:P
        if d==D+1 % ------------------------------------------- k_iSE and k_jSE
          ii_ = ii2; ij_ = ij2;
          eXij = exp(-2*X2(1:D,i)) + exp(-2*X2(1:D,j));
          ki = k(:,i,D+1); kj = k(:,j,D+1);
        elseif e==1 % ------------------------------------------ k_id and k_jSE
          ii_ = 0*ii1; ii_(:,d) = ii1(:,d); ij_ = ij2;
          eXij = exp(-2*X2(1:D,j)); eXij(d) = eXij(d) + exp(-2*X1(d,i));
          ki = k(:,i,d); kj = k(:,j,D+1);
        else % ------------------------------------------------- k_iSE and k_jd
          ij_= 0*ij1; ij_(:,d) = ij1(:,d); ii_ = ii2;
          eXij = exp(-2*X2(1:D,i)); eXij(d) = eXij(d) + exp(-2*X1(d,j));
          kj = k(:,j,d); ki = k(:,i,D+1);
        end
        R = s*diag(eXij) + eye(D);
        t = 1/sqrt(det(R));
        L = exp(bsxfun(@plus,ki,kj') + maha(ii_,-ij_,R\s/2));
        LL = LL + pp*t*L;
      end % e
    end % d
    
    S(i,j) = sum(sum(BB.*LL)); S(j,i) = S(i,j);          % predicted covariance
    
  end % j
  S(i,i) = S(i,i) + sum(exp(2*X(D2+1:D3+1,i)));
end % i

S = S - M*M';                                              % centralize moments