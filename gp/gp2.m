function [M, S, V] = gp2(gpmodel, m, s)

% Compute joint predictions for multiple GPs with uncertain inputs. If
% dynmodel.nigp exists, individual noise contributions are added. Predictive
% variances contain uncertainty about the function, but no noise.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters                                    [ P  x  E ]
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
% Copyright (C) 2008-2012 by Carl Edward Rasmussen & Marc Deisenroth
% 2015-03-15

persistent iK oldX oldIn oldOut beta oldn;
D = size(gpmodel.inputs,2);    % number of examples and dimension of inputs
[n, E] = size(gpmodel.target);      % number of examples and number of outputs

input = gpmodel.inputs;  target = gpmodel.target; X = gpmodel.hyp;

% if necessary: re-compute cashed variables
if numel(X) ~= numel(oldX) || isempty(iK) ||  n ~= oldn || ...
        sum(any(X ~= oldX)) || sum(any(oldIn ~= input)) || ...
        sum(any(oldOut ~= target))
    oldX = X; oldIn = input; oldOut = target; oldn = n;
    K = zeros(n,n,E); iK = K; beta = zeros(n,E);
    
    for i=1:E                                              % compute K and inv(K)
        inp = bsxfun(@rdivide,gpmodel.inputs,exp(X(1:D,i)'));
        K(:,:,i) = exp(2*X(D+1,i)-maha(inp,inp)/2);
        if isfield(gpmodel,'nigp')
            L = chol(K(:,:,i) + exp(2*X(D+2,i))*eye(n) + diag(gpmodel.nigp(:,i)))';
        else
            L = chol(K(:,:,i) + exp(2*X(D+2,i))*eye(n))';
        end
        iK(:,:,i) = L'\(L\eye(n));
        beta(:,i) = L'\(L\gpmodel.target(:,i));
    end
end

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);

inp = bsxfun(@minus,gpmodel.inputs,m');                    % centralize inputs

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
    L = diag(exp(-X(1:D,i)));
    in = inp*L;
    B = L*s*L+eye(D);
    
    iniB = in/B;
    l = exp(-sum(in.*iniB,2)/2); lb = l.*beta(:,i);
    t = iniB*L;
    c = exp(2*X(D+1,i))/sqrt(det(B));
    
    M(i) = sum(lb)*c;                                            % predicted mean
    V(:,i) = t'*lb*c;                   % inv(s) times input-output covariance
    k(:,i) = 2*X(D+1,i)-sum(in.*in,2)/2;
end

for i=1:E                  % compute predictive covariance, non-central moments
    ii = bsxfun(@rdivide,inp,exp(2*X(1:D,i)'));
    
    for j=1:i
        R = s*diag(exp(-2*X(1:D,i))+exp(-2*X(1:D,j)))+eye(D);
        t = 1/sqrt(det(R));
        ij = bsxfun(@times,inp,exp(-2*X(1:D,j)'));
        L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
        S(i,j) = beta(:,i)'*L*beta(:,j)*t; S(j,i) = S(i,j);
    end
    
    %   S(i,i) = S(i,i) + exp(2*X(D+1,i));
    S(i,i) = S(i,i) + 1e-6;          % add small jitter for numerical reasons
    
end

S = S - M*M';                                              % centralize moments
