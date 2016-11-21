function [M, S, V] = gpa1(gpmodel, m, s)

% Compute joint predictions for the FITC sparse approximation to multiple GPs
% with first order additive squared exponential kernels and uncertain inputs.
% If dynmodel.nigp exists, individual noise contributions are added.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters ( P = 2*D+1 )                      [ P  x  E ]
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   induce  inducint inputs                                         [ np x  D ]
%   nigp    (optional) individual noise variance terms              [ n  x  E ]
% m         mean of the test distribution                           [ D       ]
% s         covariance matrix of the test distribution              [ D  x  D ]
%
% M         mean of pred. distribution                              [ E       ]
% S         covariance of the pred. distribution                    [ E  x  E ]
% V         inv(s) times covariance between input and output        [ D  x  E ]
%
% Copyright (C) 2008-2012 by Marc Deisenroth & Carl Edward Rasmussen 2012-01-15

if numel(gpmodel.induce)==0, [M, S, V] = gpa0(gpmodel, m, s); return; end

persistent iK iK2 beta oldX;
ridge = 1e-6;                        % jitter to make matrix better conditioned
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
E = size(gpmodel.target,2);         % number of examples and number of outputs
X = gpmodel.hyp; input = gpmodel.inputs; target = gpmodel.target;
D2 = D*2;

[np pD pE] = size(gpmodel.induce);     % number of pseudo inputs per dimension
pinput = gpmodel.induce;                                   % all pseudo inputs

if numel(X) ~= numel(oldX) || isempty(iK) || isempty(iK2) || ... % if necessary
              sum(any(X ~= oldX)) || numel(iK2) ~=E*np^2 || numel(iK) ~= n*np*E
  oldX = X;                                        % compute K, inv(K), inv(K2)
  iK = zeros(np,n,E); iK2 = zeros(np,np,E); beta = zeros(np,E);
    
  for i=1:E
    pinp = bsxfun(@rdivide,pinput(:,:,min(i,pE)),exp(X(1:D,i)'));
    inp = bsxfun(@rdivide,input,exp(X(1:D,i)'));
    Kmm = ridge*eye(np); Kmn = zeros(np,n);                   % add small ridge
    for d=1:D
      Kmm = Kmm + exp(2*X(D+d,i)-maha(pinp(:,d),pinp(:,d))/2);
      Kmn = Kmn + exp(2*X(D+d,i)-maha(pinp(:,d), inp(:,d))/2);
    end
    L = chol(Kmm)';
    V = L\Kmn;                                             % inv(sqrt(Kmm))*Kmn
    if isfield(gpmodel,'nigp')
      G = sum(exp(2*X(D+1:D2,i))) - sum(V.^2) + gpmodel.nigp(:,i)';
    else
      G = sum(exp(2*X(D+1:D2,i))) - sum(V.^2);
    end
    G = sqrt(1+G/exp(2*X(D2+1,i)));
    V = bsxfun(@rdivide,V,G);
    Am = chol(exp(2*X(D2+1,i))*eye(np) + V*V')';
    At = L*Am;                                    % chol(sig*B) [thesis, p. 40]
    iAt = At\eye(np);
% The following is not an inverse matrix, but we'll treat it as such: multiply
% the targets from right and the cross-covariances left to get predictive mean.
    iK(:,:,i) = ((Am\(bsxfun(@rdivide,V,G)))'*iAt)';
    beta(:,i) = iK(:,:,i)*target(:,i);      
    iB = iAt'*iAt.*exp(2*X(D2+1,i));             % inv(B), [Ed's thesis, p. 40]
    iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
  end
end

k = zeros(np,E,D); M = zeros(E,1); V = zeros(D,E); S = zeros(E);     % allocate
inp = zeros(np,D,E);

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  inp(:,:,i) = bsxfun(@minus,pinput(:,:,min(i,pE)),m');
  
  for d=1:D
    L = exp(-X(d,i));
    in = inp(:,d,i)*L;
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
  ii = bsxfun(@rdivide,inp(:,:,i),exp(2*X(1:D,i)'));
  
  for j=1:i
    ij = bsxfun(@rdivide,inp(:,:,j),exp(2*X(1:D,j)'));    
    BB = beta(:,i)*beta(:,j)';
    if i==j; BB = BB - iK2(:,:,i); end          % incorporate model uncertainty

    for d=1:D
      for e=1:D
        if d==e
          sde = s(d,d); p = 1;
          ii_d = ii(:,d); ij_e = ij(:,e);
          eXi = exp(-2*X(d,i)); eXj = exp(-2*X(e,j));
        else
          sde = s([d e],[d e]); p = 2;
          ii_d = [ii(:,d) zeros(n,1)]; ij_e = [zeros(n,1) ij(:,e)];
          eXi = [exp(-2*X(d,i)) 0]; eXj = [0 exp(-2*X(e,j))];
        end
        R = sde*diag(eXi + eXj) + eye(p);
        t = 1/sqrt(det(R));
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') + maha(ii_d,-ij_e,R\sde/2));
        ssA = sum(sum(BB.*L));
        S(i,j) = S(i,j) + t*ssA; S(j,i) = S(i,j);        % predicted covariance
      end % e
    end % d  
  end % j
  S(i,i) = S(i,i) + sum(exp(2*X(D+1:D2,i)));
end % i

S = S - M*M';                                              % centralize moments