function [M, S, C] = conGaussAdd(policy, m, s)
% Compute joint predictions for multiple Radial Basis Functions (RBFs) with
% first order additive squared exponential kernels and uncertain inputs.
%
% policy  	policy struct
%   .p      parameters being optimized
%     .cen  basis function centres                                      [n x D]
%     .ldw  log sqrt weights for each dimension                         [D x E]
%     .ll   log lengthscales (1 per input dim)                          [D x E] 
%     .w    basis function weights                                      [n x E]
%
% m         mean of the test distribution                               [D    ]
% s         covariance matrix of the test distribution                  [D x D]
%
% M         mean of control distribution                                [E x 1]
% S         covariance of the control distribution                      [E x E]
% C         inv(s) times covariance between input and controls          [D x E]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen & Marc Deisenroth 2012-01-81
% Edited by Joe Hall and Andrew McHutchon 2012-06-26

ll = policy.p.ll; w = policy.p.w; 
if isfield(policy,'cen'); cen = policy.c; else cen = policy.p.c; end

[n, D] = size(cen);                % number of examples and dimension of inputs
E = size(w,2);                       % number of examples and number of outputs

% initialisations
M = zeros(E,1); S = zeros(E); C = zeros(D,E); k = zeros(n,E,D);
ii_ = zeros(n,2); ij_ = zeros(n,2); sde = zeros(2); eXi = zeros(1,2);
eXj = zeros(1,2); R = zeros(2); iR = zeros(2);

inp = bsxfun(@minus,cen,m');                % centralize basis function centres
for i=1:E       % compute control mean and inv(s) times input-output covariance
  for d=1:D
    il = exp(-ll(d,i));
    in = inp(:,d)*il;
    B = il*s(d,d)*il + 1;
    tt = in/B;
    l = exp( 2*ldw(d,i) - in.*tt/2)/sqrt(B);
    lb = l.*w(:,i); tL = tt*il;
    
    M(i) = M(i) + sum(lb);                                       % control mean
    C(d,i) = tL'*lb;                     % inv(s) times input-output covariance
    k(:,i,d) = 2*ldw(d,i) - in.*in/2;
  end
end

for i=1:E                     % compute control covariance, non-central moments
  ii = bsxfun(@rdivide,inp,exp(2*ll(:,i)));
  
  for j=1:i
    ij = bsxfun(@rdivide,inp,exp(2*ll(:,j)));
    
    for d=1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end
      for e = 1:P
        if d==e % ----------------- combo of 1D kernels on same input dimension
          p = 1; pp = 1;                                       %     d==e:  p=1
          ii_(:,1) = ii(:,d); ij_(:,1) = ij(:,e);              %  ii_/ij_   [n]
          sde(1,1) = s(d,d);                                   %      sde   [1]
          eXi(1) = exp(-2*ll(d,i)); eXj(1) = exp(-2*ll(e,j));  % l-scales   [1]
          R(1,1) = sde(1,1)*diag(eXi(1)+eXj(1))+1; iR(1,1) = 1/R(1,1);% R [pxp]
        else % -------------- combo of 1D kernels on different input dimensions
          p = 2;                                               %     d~=e:  p=2
          ii_= [ii(:,d) zeros(n,1)]; ij_= [zeros(n,1) ij(:,e)];%  ii_/ij_ [nx2]
          sde = s([d e],[d e]);                                %      sde [2x2]
          eXi = [exp(-2*ll(d,i)) 0]; eXj = [0 exp(-2*ll(e,j))];% l-scales [1x2]
          R = sde*diag(eXi+eXj)+eye(2); iR=R\eye(2);           %        R [pxp]
        end
        t = 1/sqrt(det(R(1:p,1:p)));                              %     t   [1]
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') ...              %     L [nxn]
                 + maha(ii_(:,1:p),-ij_(:,1:p),R(1:p,1:p)\sde(1:p,1:p)/2));
        ssA = w(:,i)'*L*w(:,j);                      %  ssA =    wi'*L*wj   [1]
        S(i,j) = S(i,j) + pp*t*ssA; S(j,i) = S(i,j);     % predicted covariance   
      end % e
    end % d      
  end % j
end % i

S = S - M*M' + 1e-6*eye(E);               % centralize moments...and add jitter